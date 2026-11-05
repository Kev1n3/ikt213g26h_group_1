import cv2
import numpy as np
import pickle
from src.capture import Capture
from src.utils.FaceDetector import FaceDetector
from src.utils.config import YUNET_MODEL_PATH, SFACE_MODEL_PATH, DATABASE_FILE
from src.utils.visualizer import draw_facebox_debug


def main():
    # if we read the models normally, it wont work on systems with æøå in the path name
    # since we are too proud of our norwegian letters and refuse to change path names,
    # we loaded the models into np arrays first and then sent them in, instead of path to the files
    empty_config = np.array([], dtype=np.uint8)
    yunet_detector = cv2.FaceDetectorYN.create(
        "onnx", np.fromfile(YUNET_MODEL_PATH, dtype=np.uint8), empty_config, (320, 320)
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

    while True:
        frame_img = cam.read_frame(target_width=640)
        if frame_img is None:
            break

        faces = face_detector.detect_faces(frame_img)

        if faces is not None:
            for face in faces:
                # Align the detected face patch using 5 keypoint and extract its 128D feature vector
                aligned_face = sface_recognizer.alignCrop(frame_img, face)

                # person -> score (highest first)
                person_scores = face_detector.get_persons_from_face(aligned_face, candidate_threshold)

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

        cv2.imshow("Face Recognition", frame_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break # Press 'q' to exit window
    cam.close()

if __name__ == "__main__":
    main()