import cv2
import numpy as np
import pickle

from FaceTracker import FaceTracker
from src.capture import Capture
from src.utils.FaceDetector import FaceDetector
from utils.anti_spoofing import AntiSpoofing
from src.utils.config import YUNET_MODEL_PATH, SFACE_MODEL_PATH, DATABASE_FILE, MINIFASNET_MODEL_PATH
from src.utils.visualizer import draw_tracked_faces
from utils.Detection import Detection


def main():
    # if we read the models normally, it wont work on systems with æøå in the path name
    # since we are too proud of our norwegian letters and refuse to change path names,
    # we loaded the models into np arrays first and then sent them in, instead of path to the files
    empty_config = np.array([], dtype=np.uint8)
    yunet_detector = cv2.FaceDetectorYN.create(
        "onnx", np.fromfile(YUNET_MODEL_PATH, dtype=np.uint8), empty_config, (320, 320), score_threshold=0.65, nms_threshold=0.2 # use np to read file instead, since it supports æøå
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
    anti_spoofing = AntiSpoofing(MINIFASNET_MODEL_PATH)

    while True:
        frame_img = cam.read_frame(target_width=1080)
        if frame_img is None:
            break

        faces = face_detector.detect_faces(frame_img)
        detections = []

        if faces is not None:
            for face in faces:
                # Align the detected face patch using 5 keypoint and extract its 128D feature vector
                aligned_face = sface_recognizer.alignCrop(frame_img, face)

                # map<person, score>
                person_scores = face_detector.get_persons_from_face(aligned_face, candidate_threshold)
                crop = anti_spoofing.crop_face(frame_img, face)

                result = anti_spoofing.predict(frame_img, face)

                if crop is not None:
                    preview = cv2.resize(crop, (320, 320))
                    cv2.imshow("MiniFASNet Crop", preview)

                (x0, y0, width, height) = face[:4]
                detections.append(Detection(
                    np.array([x0 + width / 2, y0 + height / 2]), # center pos
                    np.array([width, height]),
                    person_scores,
                    result
                ))

        face_tracker.process_raw_detections(detections)

        # Draws boxes with label, similarity and certainty for every tracked face
        draw_tracked_faces(frame_img, face_tracker.trackedFaces)

        cv2.imshow("Face Recognition", frame_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break # Press 'q' to exit window
    cam.close()

if __name__ == "__main__":
    main()