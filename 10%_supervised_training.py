import os
import shutil
import random
from ultralytics import YOLO

###############
# Configuration
###############

DATASET_DIR = "C:\Shrey\CAR\Car"
TEMP_DIR = "C:\Shrey\CAR\Temp"

SPLIT_RATIO = 0.10
SEED = 42

NC = 1  # Number of classes
CLASS_NAMES = ["car"]  # Class names
EPOCHS = 25  # Number of training epochs
IMGSZ = 640  # Image size for training
BATCH_SIZE = 16  # Batch size for training

def create_split(dataset_dir, temp_dir, ratio, seed):
    random.seed(seed)

    for split in ["train", "valid", "test"]:
        src_images = os.path.join(dataset_dir, split, "images")
        src_labels = os.path.join(dataset_dir, split, "labels")

        dst_images = os.path.join(temp_dir, split, "images")
        dst_labels = os.path.join(temp_dir, split, "labels")
        os.makedirs(dst_images, exist_ok=True)
        os.makedirs(dst_labels, exist_ok=True)

        all_images = sorted([
            f for f in os.listdir(src_images)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ])

        selected = random.sample(all_images, max(1, int(ratio * len(all_images))))
        print(f"[INFO] {split}: selected {len(selected)} / {len(all_images)} images")

        for img_file in selected:
            shutil.copy(
                os.path.join(src_images, img_file),
                os.path.join(dst_images, img_file)
            )
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_src  = os.path.join(src_labels, label_file)
            if os.path.exists(label_src):
                shutil.copy(label_src, os.path.join(dst_labels, label_file))
            else:
                print(f"[WARN] No label found for {img_file}, skipping.")

def create_yaml(temp_dir, nc, names):
    yaml_path = os.path.join(temp_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {temp_dir}\n")
        f.write(f"train: train/images\n")
        f.write(f"val:   valid/images\n")
        f.write(f"test:  test/images\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {names}\n")
    print(f"[INFO] YAML written to {yaml_path}")
    return yaml_path

def train(yaml_path):
    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        project="runs/supervised",
        name="10pct_baseline",
    )
    print("[INFO] Training complete.")

def cleanup(temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"[INFO] Deleted temp folder: {temp_dir}")

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    training_successful = False
    try:
        create_split(DATASET_DIR, TEMP_DIR, SPLIT_RATIO, SEED)
        yaml_path = create_yaml(TEMP_DIR, NC, CLASS_NAMES)
        train(yaml_path)
        training_successful = True
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        cleanup(TEMP_DIR)