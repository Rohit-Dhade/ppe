"""
Unit and integration tests for PPE Detection module.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.detector import (
    BoundingBox,
    ComplianceReport,
    ComplianceStatus,
    DetectionCategory,
    DetectionItem,
    PPEDetector,
    classify_label,
)


def test_classify_label():
    """Verify classification of PPE equipment, explicit violations, workers, and other objects."""
    # Explicit violations
    assert classify_label("no_helmet") == DetectionCategory.VIOLATION
    assert classify_label("no-hardhat") == DetectionCategory.VIOLATION
    assert classify_label("NO-SAFETY VEST") == DetectionCategory.VIOLATION
    assert classify_label("no_goggle") == DetectionCategory.VIOLATION
    assert classify_label("no-gloves") == DetectionCategory.VIOLATION
    assert classify_label("no-boots") == DetectionCategory.VIOLATION
    assert classify_label("no-mask") == DetectionCategory.VIOLATION
    assert classify_label("fall-detected") == DetectionCategory.VIOLATION

    # PPE Equipment
    assert classify_label("hardhat") == DetectionCategory.PPE_EQUIPMENT
    assert classify_label("helmet") == DetectionCategory.PPE_EQUIPMENT
    assert classify_label("safety vest") == DetectionCategory.PPE_EQUIPMENT
    assert classify_label("gloves") == DetectionCategory.PPE_EQUIPMENT
    assert classify_label("goggles") == DetectionCategory.PPE_EQUIPMENT
    assert classify_label("safety shoes") == DetectionCategory.PPE_EQUIPMENT
    assert classify_label("boots") == DetectionCategory.PPE_EQUIPMENT

    # Person
    assert classify_label("person") == DetectionCategory.PERSON
    assert classify_label("Worker") == DetectionCategory.PERSON

    # Other / Machinery
    assert classify_label("excavator") == DetectionCategory.OTHER
    assert classify_label("dump truck") == DetectionCategory.OTHER
    assert classify_label("ladder") == DetectionCategory.OTHER
    assert classify_label("unknown_item") == DetectionCategory.OTHER


def test_detector_missing_model():
    """Verify detector raises FileNotFoundError when weights are missing."""
    with pytest.raises(FileNotFoundError):
        PPEDetector(model_path="models/non_existent_weights.pt")


def test_detector_initialization():
    """Verify detector loads model and discovers violation classes."""
    detector = PPEDetector(model_path="models/ppe_yolov8s.pt")
    assert detector.model is not None
    assert len(detector.names) > 0
    assert len(detector.supported_violation_classes) > 0
    assert "no-hardhat" in detector.supported_violation_classes or "no-gloves" in detector.supported_violation_classes


def test_detector_predict_compliant_sample():
    """Verify detection on a compliant sample image."""
    detector = PPEDetector(model_path="models/ppe_yolov8s.pt", conf_threshold=0.25)
    sample_path = "data/samples/sample_compliant_1.jpg"
    assert Path(sample_path).exists()

    result = detector.predict(sample_path)
    assert result.original_image.ndim == 3
    assert result.annotated_image.shape == result.original_image.shape
    assert len(result.detections) > 0
    assert result.inference_time_ms > 0
    assert "hardhat" in result.class_counts or "person" in result.class_counts


def test_detector_predict_violation_sample():
    """Verify detection on an image known to have missing-PPE violations."""
    detector = PPEDetector(model_path="models/ppe_yolov8s.pt", conf_threshold=0.25)
    sample_path = "data/samples/sample_violation_1.jpg"
    assert Path(sample_path).exists()

    result = detector.predict(sample_path)
    # This sample has no-safety vest or no-hardhat
    assert result.compliance_report.status == ComplianceStatus.VIOLATION_DETECTED
    assert result.compliance_report.total_violations > 0
    assert len(result.compliance_report.violation_items) > 0


def test_compliance_evaluation_logic():
    """Verify that compliance evaluation never claims complete compliance falsely."""
    detector = PPEDetector(model_path="models/ppe_yolov8s.pt")

    # Case 1: Explicit violations present
    dets_with_violation = [
        DetectionItem(
            label="no-hardhat",
            confidence=0.85,
            bbox=BoundingBox(10, 10, 50, 50),
            category=DetectionCategory.VIOLATION,
            class_id=2,
        ),
        DetectionItem(
            label="person",
            confidence=0.90,
            bbox=BoundingBox(5, 5, 100, 200),
            category=DetectionCategory.PERSON,
            class_id=0,
        ),
    ]
    report1 = detector._evaluate_compliance(dets_with_violation)
    assert report1.status == ComplianceStatus.VIOLATION_DETECTED
    assert report1.total_violations == 1

    # Case 2: Person detected with helmet, NO explicit violations
    # MUST be marked INCONCLUSIVE, not claimed as 100% compliant
    dets_compliant = [
        DetectionItem(
            label="person",
            confidence=0.90,
            bbox=BoundingBox(5, 5, 100, 200),
            category=DetectionCategory.PERSON,
            class_id=0,
        ),
        DetectionItem(
            label="hardhat",
            confidence=0.88,
            bbox=BoundingBox(10, 10, 50, 50),
            category=DetectionCategory.PPE_EQUIPMENT,
            class_id=1,
        ),
    ]
    report2 = detector._evaluate_compliance(dets_compliant)
    assert report2.status == ComplianceStatus.INCONCLUSIVE_NO_EXPLICIT_VIOLATIONS
    assert "full individual compliance cannot be certified automatically" in report2.details.lower()

    # Case 3: Empty scene
    report3 = detector._evaluate_compliance([])
    assert report3.status == ComplianceStatus.NO_PERSON_DETECTED


def test_predict_invalid_inputs():
    """Verify detector handles invalid inputs gracefully."""
    detector = PPEDetector(model_path="models/ppe_yolov8s.pt")

    # None input
    with pytest.raises(ValueError, match="Provided image input is None"):
        detector.predict(None)

    # Empty array
    with pytest.raises(ValueError, match="empty"):
        detector.predict(np.array([]))

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        detector.predict("non_existent_image.png")

    # Corrupt bytes
    with pytest.raises(ValueError, match="Could not decode image"):
        detector.predict(b"not an image data")


def test_save_result(tmp_path):
    """Verify annotated output and JSON metadata are saved properly."""
    detector = PPEDetector(model_path="models/ppe_yolov8s.pt")
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    result = detector.predict(dummy_img)

    img_file, json_file = detector.save_result(result, output_dir=tmp_path)
    assert Path(img_file).exists()
    assert Path(json_file).exists()

    with open(json_file, "r") as f:
        data = json.load(f)
    assert "compliance_report" in data
    assert "class_counts" in data
    assert "inference_time_ms" in data
