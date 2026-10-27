"""
Reproducible Fine-Tuning Script for PPE Detection
Fine-tunes a YOLO model on the Ultralytics Construction-PPE dataset.
Preserves official train, val, and test splits.
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import torch
from ultralytics import YOLO
import yaml


def resolve_dataset_yaml(raw_yaml_path: Path, output_yaml_path: Path) -> Path:
    """
    Reads the dataset YAML and ensures the 'path' entry points to the
    absolute directory of the construction-ppe dataset.
    """
    if not raw_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {raw_yaml_path}")

    with open(raw_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Locate the construction-ppe directory
    candidate_dir = raw_yaml_path.parent / "construction-ppe"
    if not candidate_dir.exists():
        candidate_dir = raw_yaml_path.parent

    data["path"] = str(candidate_dir.resolve())

    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)

    return output_yaml_path


def train_ppe_model(
    model_name: str = "yolov8s.pt",
    data_yaml: str = "data/raw/data.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "auto",
    project: str = "runs/train",
    name: str = "ppe_yolo",
    save_to_models: bool = True,
):
    """
    Fine-tunes YOLO model on PPE dataset and optionally saves best weights to models/.
    """
    root_dir = Path(__file__).resolve().parent.parent
    raw_yaml = (root_dir / data_yaml).resolve()
    resolved_yaml = root_dir / "data" / "construction-ppe-runtime.yaml"

    print(f"Preparing dataset configuration from {raw_yaml}...")
    final_yaml_path = resolve_dataset_yaml(raw_yaml, resolved_yaml)
    print(f"Dataset config resolved to: {final_yaml_path}")

    # Determine device
    if device == "auto":
        resolved_device = 0 if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device
    print(f"Using device: {resolved_device}")

    print(f"Loading base YOLO model: {model_name}...")
    model = YOLO(model_name)

    print(f"Starting fine-tuning: {epochs} epochs, batch {batch_size}, img size {imgsz}...")
    results = model.train(
        data=str(final_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=resolved_device,
        project=project,
        name=name,
        exist_ok=True,
    )

    best_checkpoint = Path(project) / name / "weights" / "best.pt"
    if best_checkpoint.exists():
        print(f"\nTraining completed! Best weights at: {best_checkpoint}")
        if save_to_models:
            target_path = root_dir / "models" / f"{name}_best.pt"
            shutil.copy(best_checkpoint, target_path)
            print(f"Copied best model to: {target_path}")
    else:
        print("Training finished. Check runs directory for logs.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on Construction-PPE dataset")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Base model weights or architecture")
    parser.add_argument("--data", type=str, default="data/raw/data.yaml", help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or CUDA index e.g. '0'")
    parser.add_argument("--project", type=str, default="runs/train", help="Output project directory")
    parser.add_argument("--name", type=str, default="ppe_fine_tuned", help="Experiment name")
    parser.add_argument("--no-copy", dest="save_to_models", action="store_false", help="Do not copy best.pt to models/")

    args = parser.parse_args()

    train_ppe_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save_to_models=args.save_to_models,
    )


if __name__ == "__main__":
    main()
