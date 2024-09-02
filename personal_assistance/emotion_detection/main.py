import cv2
from emotion_detection.face_detection import FaceDetector
from emotion_detection.emotion_model import load_emotion_model, predict_emotion
from emotion_detection.utils import extract_face
import detected_emotion_global #import gv

def emotion_detection():

    face_detector = FaceDetector()
    emotion_model = load_emotion_model()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_landmarks = face_detector.detect_faces(frame)
        if face_landmarks:
            for landmarks in face_landmarks:
                face_img = extract_face(frame, landmarks)
                if face_img.size > 0:
                    # recognize the emotion
                    emotion_label = predict_emotion(emotion_model, face_img)
                    # update global variable
                    detected_emotion_global.detected_emotion = emotion_label

                    # show the emotion label
                    cv2.putText(frame, str(emotion_label), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Extracted Face', face_img)

            # Plot landmarks
            frame = face_detector.draw_landmarks(frame, face_landmarks)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


