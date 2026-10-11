import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

YUNET_MODEL = os.path.join(BASE_DIR, "models", "face_detection_yunet_2026may.onnx")
SFACE_MODEL = os.path.join(BASE_DIR, "models", "face_recognition_sface_2021dec.onnx")
DATABASE_FILE = os.path.join(BASE_DIR, "models", "face_db.pkl")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")