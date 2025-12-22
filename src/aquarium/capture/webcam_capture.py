import cv2

class WebcamCapture:
    def __init__(self, webcam_index: int = 0):
        self.cap = cv2.VideoCapture(webcam_index)

    def capture(self):
        ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()
