import cv2
import pickle
from src.capture import Capture
from src.utils.config import YUNET_MODEL_PATH, SFACE_MODEL_PATH, DATABASE_FILE
from src.utils.visualizer import draw_facebox


def main():
    yunet_detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (320, 320)) # face detection
    sface_recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "") # 128d feature extraction

    # Loads database containing reference images of known faces
    with open(DATABASE_FILE, "rb") as f:
        face_database = pickle.load(f)

    cam = Capture(src=0)

    # Cosine similarity threshold
    sface_threshold = 0.363

    while True:
        frame_img = cam.read_frame(target_width=640)
        if frame_img is None:
            break

        height, width = frame_img.shape[:2]
        yunet_detector.setInputSize((width, height)) # dynamically match frame size of video frame
        # faces: np.ndarray (N, 15) | None. columns: x, y, w, h, 5 face-keypoints (x, y), score
        _, faces = yunet_detector.detect(frame_img)

        if faces is not None:
            for face in faces:
                # Align the detected face patch using 5 keypoint and extract its 128D feature vector
                aligned_face = sface_recognizer.alignCrop(frame_img, face)
                face_feature = sface_recognizer.feature(aligned_face)

                best_match = "Unknown"
                max_score = 0.0

                # Compare the live face features against all stored reference picture embeddings in the d8abase
                for person_name, stored_embeddings in face_database.items():
                    for stored_feature in stored_embeddings:
                        score = sface_recognizer.match(face_feature, stored_feature, cv2.FaceRecognizerSF_FR_COSINE)
                        if score > max_score:
                            max_score = score
                            if score >= sface_threshold:
                                best_match = person_name

                is_recognized = (best_match != "Unknown")

                # Draws boxes with label and similarity score
                draw_facebox(
                    frame      = frame_img,
                    face_boxes = face,
                    name       = best_match,
                    similarity = max_score,
                    recognized = is_recognized
                )

        cv2.imshow("Face Recognition", frame_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break # Press 'q' to exit window
    cam.close()

if __name__ == "__main__":
    main()