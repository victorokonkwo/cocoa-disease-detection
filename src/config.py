import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(BASE_DIR, "src", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "raw", "Train.csv")
TEST_CSV = os.path.join(DATA_DIR, "raw", "Test.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "annotations", "dataset", "images", "train")

DEVICE = "cuda" # or cpu
MODEL_NAME = "yolov8n"  # nano version for mobile efficiency
IMG_SIZE = 640
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

CLASS_NAMES = ["healthy", "diseased"]  # Update based on actual classes

QUANTIZE = True
EXPORT_FORMAT = "onnx"  # or "tflite"

EXPLAINER_TYPE = "grad_cam"  # or "lime" or "shap"