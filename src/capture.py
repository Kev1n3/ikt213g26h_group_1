import cv2

class Capture:
    def __init__(self, src = 0):
        self.camera = cv2.VideoCapture(src)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera: {src}")

    def read_frame(self, target_width = None ):
        ret, frame = self.camera.read()
        if not ret:
            print ("Cannot read frame")
            return None

        ## Resize fram to improve face recognition FPS
        if target_width and frame is not None:
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            target_height = int(target_width / aspect_ratio)
            frame = cv2.resize(frame, (target_width, target_height))

        return frame

    def close(self):
        if self.camera.isOpened():
            self.camera.release()
            cv2.destroyAllWindows()
