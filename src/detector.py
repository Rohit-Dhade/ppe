"""
PPE Detection Module
Handles object detection, class categorization, compliance evaluation,
visual annotations, and output persistence.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class DetectionCategory(str, Enum):
    """Categorization for detected objects in safety inspection."""
    VIOLATION = "Violation"
    PPE_EQUIPMENT = "PPE Equipment"
    PERSON = "Worker"
    OTHER = "Equipment / Other"


class ComplianceStatus(str, Enum):
    """Overall safety assessment status."""
    VIOLATION_DETECTED = "VIOLATION_DETECTED"
    INCONCLUSIVE_NO_EXPLICIT_VIOLATIONS = "INCONCLUSIVE_NO_EXPLICIT_VIOLATIONS"
    NO_PERSON_DETECTED = "NO_PERSON_DETECTED"
    EMPTY_IMAGE = "EMPTY_IMAGE"


@dataclass
class BoundingBox:
    """Represents bounding box coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class DetectionItem:
    """Represents a single detected object."""
    label: str
    confidence: float
    bbox: BoundingBox
    category: DetectionCategory
    class_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "category": self.category.value,
            "class_id": int(self.class_id),
            "bbox": self.bbox.to_list(),
        }


@dataclass
class ComplianceReport:
    """Structured safety compliance assessment."""
    status: ComplianceStatus
    summary_message: str
    details: str
    total_persons: int
    total_ppe_items: int
    total_violations: int
    violation_items: List[DetectionItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "summary_message": self.summary_message,
            "details": self.details,
            "total_persons": self.total_persons,
            "total_ppe_items": self.total_ppe_items,
            "total_violations": self.total_violations,
            "violation_items": [v.to_dict() for v in self.violation_items],
        }


@dataclass
class PPEDetectionResult:
    """Comprehensive result of PPE inference on an image."""
    original_image: np.ndarray  # RGB format
    annotated_image: np.ndarray  # RGB format
    detections: List[DetectionItem]
    class_counts: Dict[str, int]
    category_counts: Dict[str, int]
    compliance_report: ComplianceReport
    inference_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inference_time_ms": round(self.inference_time_ms, 2),
            "class_counts": self.class_counts,
            "category_counts": self.category_counts,
            "compliance_report": self.compliance_report.to_dict(),
            "detections": [d.to_dict() for d in self.detections],
        }


def classify_label(label: str) -> DetectionCategory:
    """
    Classifies a label string into DetectionCategory.
    Explicitly flags missing-PPE classes (e.g. no-hardhat, no_helmet, no-gloves)
    as violations.
    """
    clean = label.strip().lower().replace("_", "-")

    # Explicit missing-PPE / violation keywords
    violation_prefixes = ("no-", "without-", "not-", "un-")
    violation_keywords = (
        "no-helmet", "no-hardhat", "no-vest", "no-safety vest",
        "no-goggle", "no-goggles", "no-glass", "no-glasses",
        "no-glove", "no-gloves", "no-boot", "no-boots",
        "no-shoes", "no-safety shoes", "no-mask",
        "no-ear-protection", "fall-detected"
    )

    if any(clean.startswith(p) for p in violation_prefixes) or any(k in clean for k in violation_keywords):
        return DetectionCategory.VIOLATION

    # Person / Worker keywords
    person_keywords = ("person", "worker", "worker-man", "people", "human")
    if clean in person_keywords:
        return DetectionCategory.PERSON

    # Standard PPE items
    ppe_keywords = (
        "helmet", "hardhat", "vest", "safety vest", "glove", "gloves",
        "goggle", "goggles", "glass", "glasses", "boot", "boots",
        "safety shoe", "safety shoes", "shoes", "mask", "ear-protection", "shield"
    )
    if any(k in clean for k in ppe_keywords):
        return DetectionCategory.PPE_EQUIPMENT

    return DetectionCategory.OTHER


# Distinct curated color palette (RGB)
CATEGORY_COLORS: Dict[DetectionCategory, Tuple[int, int, int]] = {
    DetectionCategory.VIOLATION: (239, 68, 68),       # Bright Crimson Red
    DetectionCategory.PPE_EQUIPMENT: (16, 185, 129),   # Emerald Green
    DetectionCategory.PERSON: (59, 130, 246),         # Royal Blue
    DetectionCategory.OTHER: (245, 158, 11),          # Amber / Orange
}


class PPEDetector:
    """
    Inference and annotation engine for PPE object detection.
    Encapsulates YOLO model loading, input validation, object detection,
    and visual rendering.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/ppe_yolov8s.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize detector with specified model weights.

        Args:
            model_path: Path to the fine-tuned YOLO .pt model.
            conf_threshold: Minimum confidence score to accept detection.
            iou_threshold: Intersection over Union threshold for NMS.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: '{self.model_path}'. "
                f"Ensure a valid fine-tuned YOLO weights file exists in 'models/'."
            )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        try:
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{self.model_path}': {str(e)}") from e

        # Extract model class names
        self.names: Dict[int, str] = getattr(self.model, "names", {})
        self.supported_violation_classes: List[str] = [
            name for name in self.names.values()
            if classify_label(name) == DetectionCategory.VIOLATION
        ]

    def _convert_input_to_rgb_array(self, image_input: Any) -> np.ndarray:
        """
        Validates and converts various image inputs (np.ndarray, PIL.Image, file path, bytes)
        into a valid RGB numpy array.
        """
        if image_input is None:
            raise ValueError("Provided image input is None.")

        if isinstance(image_input, Image.Image):
            rgb_img = np.array(image_input.convert("RGB"))
            if rgb_img.size == 0:
                raise ValueError("PIL Image is empty.")
            return rgb_img

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Could not decode image from file: {path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Could not decode image from raw bytes.")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Numpy image array is empty.")
            # If 2D grayscale, convert to 3D RGB
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            if image_input.ndim == 3:
                # Ensure 3 channels
                if image_input.shape[2] == 4:
                    return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                if image_input.shape[2] == 3:
                    return image_input.copy()
            raise ValueError(f"Invalid image array shape: {image_input.shape}")

        # Check for stream/file-like object (e.g. Streamlit UploadedFile)
        if hasattr(image_input, "read"):
            data = image_input.read()
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            return self._convert_input_to_rgb_array(data)

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _evaluate_compliance(
        self,
        detections: List[DetectionItem],
    ) -> ComplianceReport:
        """
        Evaluates overall safety compliance based on detected items.
        Strictly prevents claiming complete compliance simply because one helmet was detected,
        honestly distinguishing detected violations from inconclusive cases.
        """
        persons = [d for d in detections if d.category == DetectionCategory.PERSON]
        violations = [d for d in detections if d.category == DetectionCategory.VIOLATION]
        ppe_items = [d for d in detections if d.category == DetectionCategory.PPE_EQUIPMENT]

        total_persons = len(persons)
        total_violations = len(violations)
        total_ppe = len(ppe_items)

        if total_violations > 0:
            violation_names = ", ".join(f"{v.label} ({v.confidence:.0%})" for v in violations)
            return ComplianceReport(
                status=ComplianceStatus.VIOLATION_DETECTED,
                summary_message=f"VIOLATIONS DETECTED: {total_violations} safety violation(s) identified.",
                details=(
                    f"Identified missing protective gear: {violation_names}. "
                    "Immediate site supervisor intervention required."
                ),
                total_persons=total_persons,
                total_ppe_items=total_ppe,
                total_violations=total_violations,
                violation_items=violations,
            )

        if total_persons > 0:
            return ComplianceReport(
                status=ComplianceStatus.INCONCLUSIVE_NO_EXPLICIT_VIOLATIONS,
                summary_message="INCONCLUSIVE: No explicit PPE violations detected.",
                details=(
                    f"Detected {total_persons} worker(s) and {total_ppe} protective item(s). "
                    "No missing-PPE flags were triggered. However, full individual compliance "
                    "cannot be certified automatically from camera angle/occlusion without human inspection."
                ),
                total_persons=total_persons,
                total_ppe_items=total_ppe,
                total_violations=0,
                violation_items=[],
            )

        if total_ppe > 0:
            return ComplianceReport(
                status=ComplianceStatus.INCONCLUSIVE_NO_EXPLICIT_VIOLATIONS,
                summary_message="INCONCLUSIVE: PPE detected without visible workers.",
                details=(
                    f"Detected {total_ppe} protective equipment items, but no person bodies were explicitly recognized."
                ),
                total_persons=0,
                total_ppe_items=total_ppe,
                total_violations=0,
                violation_items=[],
            )

        return ComplianceReport(
            status=ComplianceStatus.NO_PERSON_DETECTED,
            summary_message="NO PERSONS OR PPE DETECTED.",
            details="No workers, protective equipment, or safety violations were detected in this image.",
            total_persons=0,
            total_ppe_items=0,
            total_violations=0,
            violation_items=[],
        )

    def _draw_annotations(
        self,
        image_rgb: np.ndarray,
        detections: List[DetectionItem],
        show_conf: bool = True,
        show_labels: bool = True,
        line_thickness: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draws clear, professional bounding boxes and labels on an RGB image.
        Uses OpenCV with clean badge backgrounds.
        """
        canvas = image_rgb.copy()
        h, w = canvas.shape[:2]

        # Calculate adaptive line thickness and font scale based on image dimensions
        thickness = line_thickness or max(2, int(round(min(w, h) / 350)))
        font_scale = max(0.45, min(w, h) / 1200.0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw in order: OTHER -> PERSON -> PPE_EQUIPMENT -> VIOLATION (violations on top)
        sort_order = {
            DetectionCategory.OTHER: 1,
            DetectionCategory.PERSON: 2,
            DetectionCategory.PPE_EQUIPMENT: 3,
            DetectionCategory.VIOLATION: 4,
        }
        sorted_dets = sorted(detections, key=lambda d: sort_order.get(d.category, 0))

        for item in sorted_dets:
            color = CATEGORY_COLORS.get(item.category, (200, 200, 200))
            # OpenCV expects BGR for drawing if using standard functions, but canvas is RGB,
            # so we use RGB tuple directly when canvas is RGB.
            box = item.bbox

            # Clamp coordinates to canvas bounds
            x1 = max(0, min(box.x1, w - 1))
            y1 = max(0, min(box.y1, h - 1))
            x2 = max(0, min(box.x2, w - 1))
            y2 = max(0, min(box.y2, h - 1))

            # Draw outer bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            # Build label text
            parts = []
            if show_labels:
                parts.append(item.label)
            if show_conf:
                parts.append(f"{item.confidence:.0%}")
            label_text = " ".join(parts)

            if label_text:
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, 1)
                # Badge background placement
                by1 = max(0, y1 - th - baseline - 6)
                by2 = y1
                bx2 = min(w, x1 + tw + 8)

                if y1 - th - baseline - 6 < 0:
                    # Place inside top of box if outside image boundary
                    by1 = y1
                    by2 = y1 + th + baseline + 6

                # Filled badge
                cv2.rectangle(canvas, (x1, by1), (bx2, by2), color, -1)
                # White text for high contrast
                text_y = by2 - baseline - 2
                cv2.putText(
                    canvas,
                    label_text,
                    (x1 + 4, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    max(1, int(thickness / 2)),
                    lineType=cv2.LINE_AA,
                )

        return canvas

    def predict(
        self,
        image_input: Any,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        show_conf: bool = True,
        show_labels: bool = True,
    ) -> PPEDetectionResult:
        """
        Runs object detection on the provided image and produces structured results.

        Args:
            image_input: Image in PIL, numpy RGB, file path, bytes, or file stream.
            conf_threshold: Optional confidence threshold override.
            iou_threshold: Optional IoU threshold override.
            show_conf: Include confidence percentages on drawn badges.
            show_labels: Include class names on drawn badges.

        Returns:
            PPEDetectionResult containing original and annotated images, detection list,
            counts by class and category, and a compliance assessment report.
        """
        rgb_image = self._convert_input_to_rgb_array(image_input)

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        # Run inference using Ultralytics YOLO
        start_time = datetime.now()
        results = self.model(
            rgb_image,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000.0

        detections: List[DetectionItem] = []
        class_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {c.value: 0 for c in DetectionCategory}

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    c_id = int(classes[i])
                    raw_label = self.names.get(c_id, f"class_{c_id}")
                    score = float(confs[i])
                    b = [int(round(coord)) for coord in xyxy[i]]
                    bbox = BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3])
                    cat = classify_label(raw_label)

                    item = DetectionItem(
                        label=raw_label,
                        confidence=score,
                        bbox=bbox,
                        category=cat,
                        class_id=c_id,
                    )
                    detections.append(item)
                    class_counts[raw_label] = class_counts.get(raw_label, 0) + 1
                    category_counts[cat.value] = category_counts.get(cat.value, 0) + 1

        # Evaluate overall compliance
        compliance_report = self._evaluate_compliance(detections)

        # Generate annotated image
        annotated_image = self._draw_annotations(
            image_rgb=rgb_image,
            detections=detections,
            show_conf=show_conf,
            show_labels=show_labels,
        )

        return PPEDetectionResult(
            original_image=rgb_image,
            annotated_image=annotated_image,
            detections=detections,
            class_counts=class_counts,
            category_counts=category_counts,
            compliance_report=compliance_report,
            inference_time_ms=elapsed_ms,
        )

    def save_result(
        self,
        result: PPEDetectionResult,
        output_dir: Union[str, Path] = "outputs",
        filename_prefix: str = "ppe_annotated",
    ) -> Tuple[str, str]:
        """
        Saves annotated image and JSON detection summary to output_dir.

        Returns:
            Tuple of (saved_image_path, saved_json_path)
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"{filename_prefix}_{timestamp}.jpg"
        json_filename = f"{filename_prefix}_{timestamp}.json"

        img_filepath = out_path / img_filename
        json_filepath = out_path / json_filename

        # Convert RGB to BGR for OpenCV write
        bgr = cv2.cvtColor(result.annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_filepath), bgr)

        # Write metadata JSON
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return str(img_filepath), str(json_filepath)
