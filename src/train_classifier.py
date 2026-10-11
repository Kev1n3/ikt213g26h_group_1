import os
import cv2
import pickle
from src.utils.config import YUNET_MODEL, SFACE_MODEL, DATABASE_FILE, DATASET_DIR

def train_classifier():
    # Check if the dataset folder exist before proceeding
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset doesn't exist: {DATASET_DIR}")
        return

    # Initialize OpenCV YuNet for the face detection and SFace 128D feature extraction
    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320,320))
    recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

    database = {}

    print(f"Processing pictures in the dataset")
    # Loop through each person's dataset folder
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        database[person_name] = []

        # Loop through each person image file for the current person
        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize extremely large images
            h, w = img.shape[:2]
            max_dim = 1280
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                h, w = img.shape[:2]

            detector.setInputSize((w, h))
            _, faces = detector.detect(img)

            # Cheks if YuNet detects least one face in the image
            if faces is not None and len(faces) > 0:
                aligned_face = recognizer.alignCrop(img, faces[0])
                embedding = recognizer.feature(aligned_face)
                database[person_name].append(embedding)

    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(database, f)

    total_embeddings = sum(len(embeds) for embeds in database.values())
    print(f"Saved {total_embeddings} face embeddings for {len(database)} person(s) to '{DATABASE_FILE}'!")

train_classifier()