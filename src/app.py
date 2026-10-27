"""
PPE Detection Prototype - Streamlit Web Application
Provides an interactive computer vision interface for inspecting construction site images,
detecting personal protective equipment (PPE), identifying safety violations,
and exporting annotated results.
"""

import sys
from io import BytesIO
import os
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image
import streamlit as st

from src.detector import (
    ComplianceStatus,
    DetectionCategory,
    PPEDetector,
    PPEDetectionResult,
)



# Streamlit Page Setup
st.set_page_config(
    page_title="PPE Safety Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        padding: 1rem 0rem 0.5rem 0rem;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .badge-violation {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .badge-inconclusive {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
    }
    .badge-info {
        background-color: #E0E7FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .alert-box-violation {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 0.375rem;
        margin-bottom: 1rem;
    }
    .alert-box-inconclusive {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.375rem;
        margin-bottom: 1rem;
    }
    .alert-box-info {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.375rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_available_models() -> Dict[str, str]:
    """Scans models directory for available .pt model checkpoints."""
    model_paths: Dict[str, str] = {}
    base_dir = Path(__file__).resolve().parent.parent

    # Search in models/ and models/candidates/
    search_dirs = [base_dir / "models", base_dir / "models" / "candidates"]
    for sdir in search_dirs:
        if sdir.exists():
            for pt_file in sorted(sdir.glob("*.pt")):
                rel_path = pt_file.relative_to(base_dir)
                label = f"{pt_file.stem} ({rel_path})"
                model_paths[label] = str(pt_file)

    if not model_paths:
        # Fallback to default path
        default_model = base_dir / "models" / "ppe_yolov8s.pt"
        model_paths["default_yolov8s"] = str(default_model)

    return model_paths


def find_available_samples() -> Dict[str, str]:
    """Scans data/samples directory for sample construction images."""
    sample_paths: Dict[str, str] = {}
    base_dir = Path(__file__).resolve().parent.parent
    sample_dir = base_dir / "data" / "samples"

    if sample_dir.exists():
        for img_file in sorted(sample_dir.glob("*.[jJ][pP][gG]")) + sorted(sample_dir.glob("*.[pP][nN][gG]")):
            # Create friendly name
            name = img_file.name
            if "compliant" in name.lower():
                label = f"Compliant Worker - {name}"
            elif "violation" in name.lower():
                label = f"Safety Violation - {name}"
            else:
                label = name
            sample_paths[label] = str(img_file)

    return sample_paths


@st.cache_resource(show_spinner="Loading YOLO PPE detection model...")
def load_detector(model_path: str, conf_thresh: float, iou_thresh: float) -> Optional[PPEDetector]:
    """Cached loader for the PPEDetector model."""
    try:
        return PPEDetector(
            model_path=model_path,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
        )
    except Exception as e:
        st.error(f"Error loading model weights from '{model_path}': {e}")
        return None


def main():
    # Header
    st.markdown(
        """
        <div class="main-header">
            <span class="status-badge badge-info">AI Safety Prototype • Ultralytics YOLO</span>
            <div class="main-title">🦺 Construction PPE & Violation Detector</div>
            <div class="main-subtitle">
                Automated computer vision detection of personal protective equipment and explicit missing-PPE violations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Model & Detection Settings")

        # Model Selector
        available_models = find_available_models()
        # Default selection priority: ppe_yolov8s
        default_idx = 0
        for idx, (label, mpath) in enumerate(available_models.items()):
            if "ppe_yolov8s" in label:
                default_idx = idx
                break

        selected_model_label = st.selectbox(
            "Select PPE Model Weights",
            options=list(available_models.keys()),
            index=default_idx,
            help="Choose between available fine-tuned YOLO checkpoints.",
        )
        selected_model_path = available_models[selected_model_label]

        # Sliders
        st.subheader("Thresholds")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=0.25,
            step=0.05,
            help="Minimum confidence score required for an object detection.",
        )
        iou_threshold = st.slider(
            "IoU / NMS Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Intersection-over-Union threshold for Non-Maximum Suppression.",
        )

        # Visual Display Toggles
        st.subheader("Display Options")
        show_conf = st.checkbox("Show Confidence Scores", value=True)
        show_labels = st.checkbox("Show Class Labels", value=True)

        st.markdown("---")
        st.caption("Developed with Ultralytics YOLOv8 & Construction-PPE Dataset (AGPL-3.0).")

    # Load Detector
    detector = load_detector(
        model_path=selected_model_path,
        conf_thresh=conf_threshold,
        iou_thresh=iou_threshold,
    )

    if detector is None:
        st.stop()

    # Model Class Info in Expander
    with st.expander("ℹ️ Active Model Information & Supported Violation Classes", expanded=False):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write(f"**Weights File:** `{Path(selected_model_path).name}`")
            st.write(f"**Total Classes Recognized:** {len(detector.names)}")
            st.write("**All Classes:**")
            st.write(", ".join([f"`{n}`" for n in detector.names.values()]))
        with col_m2:
            st.write("**Explicit Missing-PPE Violation Classes:**")
            if detector.supported_violation_classes:
                st.write(", ".join([f"🔴 `{n}`" for n in detector.supported_violation_classes]))
            else:
                st.write("_None explicitly labeled as missing-PPE in this model's vocabulary._")

    # Image Input Selection
    input_mode = st.radio(
        "Select Inspection Source:",
        ["🖼️ Benchmark Construction Samples", "📁 Upload Image"],
        horizontal=True,
    )
    image_to_process = None
    image_source_name = "image"

    if input_mode == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Choose a construction site image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload an image from a jobsite or work area for PPE inspection.",
        )
        if uploaded_file is not None:
            try:
                image_to_process = Image.open(uploaded_file)
                image_source_name = uploaded_file.name
            except Exception as e:
                st.error(f"Failed to read uploaded image: {e}")
        else:
            st.info("👋 Upload a construction site image (.jpg, .jpeg, .png) to begin PPE inspection.")
            st.stop()
    else:
        sample_dict = find_available_samples()
        if sample_dict:
            selected_sample_label = st.selectbox(
                "Select a benchmark sample image:",
                options=list(sample_dict.keys()),
                index=0,
                help="Preloaded test cases including fully equipped workers and critical safety violations.",
            )
            sample_path = sample_dict[selected_sample_label]
            try:
                image_to_process = Image.open(sample_path)
                image_source_name = Path(sample_path).name
            except Exception as e:
                st.error(f"Failed to load sample image: {e}")
                st.stop()
        else:
            st.warning("No sample images found in `data/samples/`.")
            st.stop()

    if image_to_process is None:
        st.stop()


    # Execute Inference
    with st.spinner("Running PPE Detection..."):
        try:
            result: PPEDetectionResult = detector.predict(
                image_input=image_to_process,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                show_conf=show_conf,
                show_labels=show_labels,
            )
        except Exception as e:
            st.error(f"Inference error: {str(e)}")
            st.stop()

    st.markdown("---")

    # 1. Compliance Status Banner
    report = result.compliance_report
    if report.status == ComplianceStatus.VIOLATION_DETECTED:
        st.markdown(
            f"""
            <div class="alert-box-violation">
                <h3 style="color: #991B1B; margin-top: 0; margin-bottom: 0.5rem;">⚠️ SAFETY VIOLATIONS DETECTED</h3>
                <p style="color: #7F1D1D; margin-bottom: 0.25rem;"><strong>{report.summary_message}</strong></p>
                <p style="color: #991B1B; margin-bottom: 0;">{report.details}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif report.status == ComplianceStatus.INCONCLUSIVE_NO_EXPLICIT_VIOLATIONS:
        st.markdown(
            f"""
            <div class="alert-box-inconclusive">
                <h3 style="color: #92400E; margin-top: 0; margin-bottom: 0.5rem;">ℹ️ NO EXPLICIT VIOLATIONS DETECTED (INCONCLUSIVE)</h3>
                <p style="color: #78350F; margin-bottom: 0.25rem;"><strong>{report.summary_message}</strong></p>
                <p style="color: #92400E; margin-bottom: 0;">{report.details}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-box-info">
                <h3 style="color: #1E40AF; margin-top: 0; margin-bottom: 0.5rem;">ℹ️ SITE INSPECTION NOTICE</h3>
                <p style="color: #1E3A8A; margin-bottom: 0.25rem;"><strong>{report.summary_message}</strong></p>
                <p style="color: #1E40AF; margin-bottom: 0;">{report.details}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Key Metrics Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-num">{len(result.detections)}</div>
                <div class="metric-label">Total Detections</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-num" style="color: #3B82F6;">{report.total_persons}</div>
                <div class="metric-label">Workers Identified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-num" style="color: #10B981;">{report.total_ppe_items}</div>
                <div class="metric-label">PPE Items Detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-num" style="color: {'#EF4444' if report.total_violations > 0 else '#10B981'};">
                    {report.total_violations}
                </div>
                <div class="metric-label">Safety Violations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-num">{result.inference_time_ms:.1f}<span style="font-size: 0.9rem;">ms</span></div>
                <div class="metric-label">Inference Time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # 3. Visual Detection Inspection (Annotated vs Original)
    img_tab1, img_tab2 = st.tabs(["🎯 Annotated Result", "📷 Original Image"])
    with img_tab1:
        st.image(
            result.annotated_image,
            caption=f"Annotated: {image_source_name} ({len(result.detections)} objects detected)",
            use_container_width=True,
        )
    with img_tab2:
        st.image(
            result.original_image,
            caption=f"Original: {image_source_name}",
            use_container_width=True,
        )

    # 4. Action & Export Controls
    st.subheader("💾 Export & Persistence")
    act_col1, act_col2, act_col3 = st.columns(3)

    with act_col1:
        if st.button("Save Annotated Result to outputs/", type="primary", use_container_width=True):
            clean_stem = Path(image_source_name).stem
            saved_img, saved_json = detector.save_result(
                result=result,
                output_dir="outputs",
                filename_prefix=f"ppe_{clean_stem}",
            )
            st.success(f"Saved annotated image to `{saved_img}` and metadata to `{saved_json}`")

    with act_col2:
        # Direct download button for annotated image
        pil_annotated = Image.fromarray(result.annotated_image)
        buf = BytesIO()
        pil_annotated.save(buf, format="JPEG")
        st.download_button(
            label="⬇️ Download Annotated Image (JPG)",
            data=buf.getvalue(),
            file_name=f"annotated_{Path(image_source_name).stem}.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

    with act_col3:
        # Download detection report as CSV
        if result.detections:
            df_export = pd.DataFrame([d.to_dict() for d in result.detections])
            # Flatten bbox for CSV
            df_export["bbox_x1"] = df_export["bbox"].apply(lambda b: b[0])
            df_export["bbox_y1"] = df_export["bbox"].apply(lambda b: b[1])
            df_export["bbox_x2"] = df_export["bbox"].apply(lambda b: b[2])
            df_export["bbox_y2"] = df_export["bbox"].apply(lambda b: b[3])
            df_export.drop(columns=["bbox"], inplace=True)
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Detections (CSV)",
                data=csv_data,
                file_name=f"detections_{Path(image_source_name).stem}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("No Detections to Export", disabled=True, use_container_width=True)

    st.markdown("---")

    # 5. Class Counts & Breakdown
    st.subheader("📊 Detected Class Breakdown & Counts")
    breakdown_col1, breakdown_col2 = st.columns([1, 1])

    with breakdown_col1:
        st.markdown("**Counts by Detected Class:**")
        if result.class_counts:
            df_counts = pd.DataFrame(
                list(result.class_counts.items()),
                columns=["Class", "Count"],
            ).sort_values(by="Count", ascending=False)
            st.dataframe(df_counts, hide_index=True, use_container_width=True)
        else:
            st.write("No objects detected at current threshold.")

    with breakdown_col2:
        st.markdown("**Counts by Category:**")
        cat_data = [
            {"Category": cat, "Count": count}
            for cat, count in result.category_counts.items()
            if count > 0
        ]
        if cat_data:
            df_cat = pd.DataFrame(cat_data).sort_values(by="Count", ascending=False)
            st.dataframe(df_cat, hide_index=True, use_container_width=True)
        else:
            st.write("No category counts to show.")

    # 6. Detailed Detection Objects Table
    st.subheader("📋 Detected Objects Detail Table")
    if result.detections:
        table_rows = []
        for idx, d in enumerate(result.detections, 1):
            table_rows.append({
                "#": idx,
                "Class Label": d.label,
                "Category": d.category.value,
                "Confidence": f"{d.confidence:.1%}",
                "Bounding Box [x1, y1, x2, y2]": f"[{d.bbox.x1}, {d.bbox.y1}, {d.bbox.x2}, {d.bbox.y2}]",
                "Box Area (px)": d.bbox.area,
            })
        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, hide_index=True, use_container_width=True)
    else:
        st.info("No detected objects to display in table.")


if __name__ == "__main__":
    main()
