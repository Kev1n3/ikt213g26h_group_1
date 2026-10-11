import cv2
import pickle
from src.capture import Capture
from src.utils.config import YUNET_MODEL, SFACE_MODEL, DATABASE_FILE
from src.utils.visualizer import draw_face_box


def main():
    # Initialize OpenCV YuNet for the face detection and SFace 128D feature extraction
    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320))
    recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

    # Loads reference database containing the known faces embeddings
    with open(DATABASE_FILE, "rb") as f:
        database = pickle.load(f)

    cam = Capture(src=0)

    # Cosine similarity threshold for SFace
    threshold = 0.363

    while True:
        frame = cam.read_frame(target_width=640)
        if frame is None:
            break

        # Dynamically update detector input dimensions to match the current frame size
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        # Process detected faces if any are present in the frame
        if faces is not None:
            for face in faces:
                # Align the detected face patch using 5 keypoint and extract its 128D feature vector
                aligned_face = recognizer.alignCrop(frame, face)
                face_feature = recognizer.feature(aligned_face)

                best_match = "Unknown"
                max_score = 0.0

                # Compare the live face feature against all stored reference picture embeddings in the databse
                for person_name, stored_embeddings in database.items():
                    for stored_feature in stored_embeddings:
                        score = recognizer.match(face_feature, stored_feature, cv2.FaceRecognizerSF_FR_COSINE)
                        if score > max_score:
                            max_score = score
                            if score >= threshold:
                                best_match = person_name

                is_recognized = (best_match != "Unknown")

                # Draws boxes with label and similarity sore
                draw_face_box(
                    frame=frame,
                    face_boxes=face,
                    name=best_match,
                    similarity=max_score,
                    recognized=is_recognized
                )

        cv2.imshow("Face Recognition", frame)

        # Press 'q' to exit window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.close()

if __name__ == "__main__":
    main()