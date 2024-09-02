import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh()

    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        return result.multi_face_landmarks

    def draw_landmarks(self, frame, face_landmarks):
        if face_landmarks:
            for landmarks in face_landmarks:
                for lm in landmarks.landmark:
                    ih, iw, _ = frame.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame
