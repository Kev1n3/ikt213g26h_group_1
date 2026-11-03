import cv2
import numpy as np
import pickle
from src.capture import Capture
from src.utils.FaceDetector import FaceDetector
from src.utils.config import YUNET_MODEL_PATH, SFACE_MODEL_PATH, DATABASE_FILE
from src.utils.visualizer import draw_facebox


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

    # Cosine similarity threshold
    sface_threshold = 0.363

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

                best_match, score = face_detector.get_person_from_face(aligned_face, sface_threshold)

                is_recognized = (best_match != "Unknown")

                # Draws boxes with label and similarity score
                draw_facebox(
                    frame      = frame_img,
                    face_boxes = face,
                    name       = best_match,
                    similarity = score,
                    recognized = is_recognized
                )

        cv2.imshow("Face Recognition", frame_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break # Press 'q' to exit window
    cam.close()

if __name__ == "__main__":
    main()