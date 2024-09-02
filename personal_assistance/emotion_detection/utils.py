import cv2

# extract the face and return the grayscale images
def extract_face(frame, landmarks):
    ih, iw, _ = frame.shape
    x1, y1 = int(min([lm.x for lm in landmarks.landmark]) * iw), int(min([lm.y for lm in landmarks.landmark]) * ih)
    x2, y2 = int(max([lm.x for lm in landmarks.landmark]) * iw), int(max([lm.y for lm in landmarks.landmark]) * ih)
    face_img = frame[y1:y2, x1:x2]
    gray_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return gray_face_img
