import cv2
import numpy as np
import pickle
import math

from FaceTracker import FaceTracker
from src.capture import Capture
from src.utils.FaceDetector import FaceDetector
from src.utils.config import YUNET_MODEL_PATH, SFACE_MODEL_PATH, DATABASE_FILE
from src.utils.visualizer import draw_facebox_debug
from utils.Detection import Detection


def main():
    empty_config = np.array([], dtype=np.uint8)
    yunet_detector = cv2.FaceDetectorYN.create(
        "onnx", np.fromfile(YUNET_MODEL_PATH, dtype=np.uint8), empty_config, (320, 320) # use np to read file instead, since it supports æøå
    ) # face detection
    sface_recognizer = cv2.FaceRecognizerSF.create(
        "onnx", np.fromfile(SFACE_MODEL_PATH, dtype=np.uint8), empty_config
    ) # 128d feature extraction

    # Loads database containing reference images of known faces
    with open(DATABASE_FILE, "rb") as f:
        face_database = pickle.load(f)

    cam = Capture(src=0)

    # Cosine similarity thresholds
    sface_threshold = 0.363 # >= 0.363 -> recognized status, will get green frame
    candidate_threshold = 0.15 # > 0.15 -> show in debug (and get displayed in frame, but not green tho)

    face_detector = FaceDetector(yunet_detector, sface_recognizer, face_database)
    
    face_tracker = FaceTracker(sface_threshold, candidate_threshold)

    while True:
        frame_img = cam.read_frame(target_width=640)
        if frame_img is None:
            break

        faces = face_detector.detect_faces(frame_img)
        detections = []

        if faces is not None:
            for face in faces:
                # Align the detected face patch using 5 keypoint and extract its 128D feature vector
                aligned_face = sface_recognizer.alignCrop(frame_img, face)

                # person -> score (highest first)
                person_scores = face_detector.get_persons_from_face(aligned_face, candidate_threshold)

                (x0, y0, Δx, Δy) = face[:4]
                detections.append(Detection(
                    np.array([x0 + Δx / 2, y0 + Δy / 2]), # pos
                    math.hypot(Δx, Δy), # diagonal size
                    person_scores
                ))

                best_match, score = next(iter(person_scores.items()), ("Unknown", 0.0)) # person_scores?.first ?: ("Unknown, 0")
                is_recognized = score >= sface_threshold
                if not is_recognized:
                    best_match = "Unknown"

                # Draws boxes with label and similarity score
                draw_facebox_debug( # also shows ppl that almost matched
                    frame         = frame_img,
                    face_boxes    = face,
                    name          = best_match,
                    similarity    = score,
                    recognized    = is_recognized,
                    person_scores = person_scores
                )

        face_tracker.process_raw_detections(detections)
                                        
        cv2.imshow("Face Recognition", frame_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break # Press 'q' to exit window
    cam.close()

if __name__ == "__main__":
    main()