"""
Train YOLOv8 on the prepared dataset using the oversampled train list.

30-epoch test run optimized for:
    NVIDIA RTX 4060 Laptop GPU - 8 GB VRAM
    16 GB RAM
    Windows

Oversampling is retained to mitigate class imbalance.
"""

from pathlib import Path
from ultralytics import YOLO
import yaml


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUT = PROJECT_ROOT / "data" / "yolo"

DATA_YAML = OUT / "data.yaml"
OVERSAMPLED_TRAIN_LIST = OUT / "train_oversampled.txt"

# Training-only YAML
TRAIN_YAML = OUT / "data_oversampled.yaml"

RUNS_DIR = PROJECT_ROOT / "runs"

RUN_NAME = "yolov8n_test30"


# ============================================================
# CREATE OVERSAMPLED TRAINING YAML
# ============================================================

base = yaml.safe_load(
    DATA_YAML.read_text()
)

if OVERSAMPLED_TRAIN_LIST.exists():

    base["train"] = OVERSAMPLED_TRAIN_LIST.as_posix()

    print(
        f"Using oversampled training list:\n"
        f"{OVERSAMPLED_TRAIN_LIST}"
    )

else:

    print(
        "WARNING: train_oversampled.txt not found."
    )

    print(
        "Using original training dataset."
    )


TRAIN_YAML.write_text(
    yaml.dump(
        base,
        sort_keys=False
    )
)

print(
    f"\nTraining YAML created:\n"
    f"{TRAIN_YAML}"
)


# ============================================================
# TRAINING
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("       YOLOv8 FAST 30-EPOCH TEST")
    print("========================================")

    print("Model        : YOLOv8n")
    print("Epochs       : 30")
    print("Image size   : 640")
    print("Batch size   : 32")
    print("Workers      : 2")
    print("Cache        : Disabled")
    print("AMP          : Enabled")
    print("Deterministic: False")
    print("Oversampling : Enabled")
    print("Device       : RTX 4060 8GB")
    print(f"Run name     : {RUN_NAME}")

    print("========================================\n")


    # ========================================================
    # LOAD PRETRAINED YOLOv8n
    # ========================================================

    model = YOLO(
        (PROJECT_ROOT / "yolov8n.pt").as_posix()
    )


    # ========================================================
    # TRAIN
    # ========================================================

    model.train(

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        data=TRAIN_YAML.as_posix(),

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        epochs=30,

        imgsz=640,

        # Your RTX 4060 should comfortably handle this
        batch=32,

        # ----------------------------------------------------
        # SPEED / MEMORY
        # ----------------------------------------------------

        # Do not cache the 42k images.
        cache=False,

        # Windows-safe
        workers=2,

        # Allow cuDNN to select faster algorithms
        deterministic=False,

        # FP16 mixed precision
        amp=True,

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        patience=20,

        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------

        device=0,

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        project=RUNS_DIR.as_posix(),

        name=RUN_NAME,

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        cls=0.7,

        box=7.5,

        dfl=1.5,

        # ----------------------------------------------------
        # CHECKPOINTS
        # ----------------------------------------------------

        save=True,

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val=True,

        verbose=True,
    )


    # ========================================================
    # TEST SET EVALUATION
    # ========================================================

    print("\n========================================")
    print("        TEST SET EVALUATION")
    print("========================================\n")

    metrics = model.val(
        data=DATA_YAML.as_posix(),
        split="test"
    )


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print("\n========================================")
    print("          FINAL TEST METRICS")
    print("========================================\n")

    print(
        "mAP50-95:",
        metrics.box.map
    )

    print(
        "mAP50:",
        metrics.box.map50
    )

    print(
        "\nFull metrics:"
    )

    print(metrics)

    print(
        "\nTraining and evaluation completed."
    )