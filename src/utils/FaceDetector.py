import cv2

class FaceDetector:
    def __init__(self, yunet_detector, sface_recognizer, face_database):
        self.yunet_detector = yunet_detector
        self.sface_recognizer = sface_recognizer
        self.face_database = face_database
        pass

    def detect_faces(self, img):
        height, width = img.shape[:2]
        self.yunet_detector.setInputSize((width, height))  # dynamically match frame size of video frame
        # faces: np.ndarray (N, 15) | None. columns: x, y, w, h, 5 face-keypoints (x, y), score
        return self.yunet_detector.detect(img)[1]

    def get_person_from_face(self, aligned_face, threshold):
        face_feature = self.sface_recognizer.feature(aligned_face)

        best_match = "Unknown"
        max_score = 0.0

        # Compare the live face features against all stored reference picture embeddings in the d8abase
        for person_name, stored_embeddings in self.face_database.items():
            for stored_feature in stored_embeddings:
                score = self.sface_recognizer.match(face_feature, stored_feature, cv2.FaceRecognizerSF_FR_COSINE)
                if score > max_score:
                    max_score = score
                    if score >= threshold:
                        best_match = person_name
        return best_match, max_score