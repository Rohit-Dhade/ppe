# Construction PPE & Safety Violation Detector

A production-grade computer vision prototype for detecting Personal Protective Equipment (PPE) and safety violations on construction sites using Ultralytics YOLOv8 and Streamlit.

---

## 📌 Project Overview

Ensuring worker safety on construction jobsites requires constant adherence to Personal Protective Equipment (PPE) mandates. This application leverages fine-tuned deep learning object detection models to:
1. Detect safety equipment (hard hats, high-visibility safety vests, protective gloves, boots/shoes, and goggles).
2. Explicitly flag missing-PPE violations (e.g. `no-hardhat`, `no-safety vest`, `no-goggles`, `no-gloves`, `no-mask`).
3. Differentiate between explicit violations and inconclusive safety states (avoiding false guarantees of compliance).
4. Provide an intuitive Streamlit interface for uploading construction imagery, inspecting detections, and exporting annotated results.

---

## 🛠️ Tech Stack

- **Computer Vision & Deep Learning**: Ultralytics YOLOv8, PyTorch
- **Image Processing**: OpenCV (headless/RGB rendering), Pillow (PIL)
- **User Interface**: Streamlit
- **Data & Testing**: Pandas, PyYAML, Pytest
- **Python Version**: Python 3.12+ (compatible with 3.9+)

---

## 🗂️ Project Structure

```text
PPE-Detector/
├── data/
│   ├── raw/
│   │   ├── construction-ppe/     # Ultralytics Construction-PPE dataset
│   │   │   ├── images/           # train (1132), val (143), test (141)
│   │   │   └── labels/           # YOLO format annotation labels
│   │   ├── data.yaml             # Dataset configuration & class definitions
│   │   └── LICENSE               # GNU AGPL-3.0 License
│   └── samples/                  # Curated benchmark sample images for quick testing
├── models/
│   ├── ppe_yolov8s.pt            # Primary fine-tuned YOLOv8s PPE model (default)
│   └── candidates/               # Additional candidate PPE model checkpoints
├── outputs/                      # Saved annotated images and detection JSON metadata
├── src/
│   ├── __init__.py
│   ├── app.py                    # Streamlit web interface
│   ├── detector.py               # Core detection engine, classification & compliance logic
│   └── train.py                  # Reproducible fine-tuning pipeline
├── tests/
│   └── test_detector.py          # Comprehensive unit & integration tests
├── requirements.txt              # Pinned Python package dependencies
├── pytest.ini                    # Pytest configuration
├── .gitignore                    # Git exclusions
└── README.md                     # Documentation
```

---

## 🛡️ Safety & Compliance Evaluation Logic

A critical requirement of safety inspection systems is **avoiding false assurances of compliance**:
- **Explicit Violation Detected (`VIOLATION_DETECTED`)**: Triggered whenever the model explicitly detects a missing-PPE class (e.g., `no-hardhat`, `no-safety vest`, `no-goggles`). Highlighted with a high-visibility crimson alert banner and detailed list of missing gear.
- **Inconclusive Case (`INCONCLUSIVE_NO_EXPLICIT_VIOLATIONS`)**: If workers and some PPE items are present but no explicit missing-PPE classes are detected, the system does **not** falsely claim that every worker is 100% compliant. Site occlusion, viewing angle, and unseen gear categories require human verification.
- **No Workers Detected (`NO_PERSON_DETECTED`)**: Displays site equipment information when no personnel are recognized.

---

## 🚀 Setup & Usage Instructions

### 1. Prerequisites & Environment Setup
Clone the repository and create a Python virtual environment:
```bash
cd PPE-Detector
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
Launch the web interface locally:
```bash
streamlit run src/app.py
```
Open your browser at `http://localhost:8501`.

### 4. Running the Application
1. **Upload an Image**: Drag-and-drop any `.jpg`, `.jpeg`, or `.png` construction image into the **📁 Upload Image** tab.
2. **Or Test with Samples**: Switch to the **🖼️ Preloaded Construction Samples** tab to instantly test compliant and violation scenarios.
3. **Adjust Settings**: Use the sidebar to tune confidence threshold, NMS IoU threshold, or toggle label displays.
4. **Export Results**:
   - Click **Save Annotated Result to outputs/** to persist the annotated image and JSON metadata.
   - Click **Download Annotated Image** or **Download Detections (CSV)** for direct exports.

---

## 🧪 Testing

The test suite covers class categorization, detector initialization, inference handling on compliant and violation images, edge cases (corrupted bytes, missing files), and output persistence:
```bash
pytest -v
```

---

## 🔄 Model Fine-Tuning (Reproducible Training Pipeline)

To retrain or fine-tune a YOLO model on the Construction-PPE dataset:
```bash
python src/train.py --model yolov8s.pt --epochs 50 --batch 16 --imgsz 640 --device auto
```
Options:
- `--model`: Base model weights or architecture (e.g. `yolov8s.pt`, `yolov8m.pt`)
- `--data`: Dataset configuration path (default: `data/raw/data.yaml`)
- `--epochs`: Number of training iterations (default: 50)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Input image size (default: 640)
- `--device`: Target device (`auto`, `cpu`, or GPU device index `0`)

The script preserves the official train (1,132 images), val (143 images), and test (141 images) splits and copies the best checkpoint to `models/`.

---

## 📜 Dataset & Model Licensing and Attribution

- **Dataset**: Ultralytics Construction-PPE dataset, downloaded from [Ultralytics Assets](https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip). Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
- **Model Checkpoint**: Fine-tuned on construction PPE detection including explicit negative violation classes. Checkpoints retain AGPL-3.0 licensing in accordance with the base Ultralytics YOLOv8 release.
- **Attribution**: Built using the [Ultralytics YOLO](https://docs.ultralytics.com/) computer vision library.
