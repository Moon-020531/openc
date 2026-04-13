import numpy as np
import os
import cv2

# Cascade 분류기 로드 함수

def load_cascade(cascade_name='haarcascade_frontalface_default.xml'):

    """Haar Cascade 분류기 로드"""

    cascade_path = cv2.data.haarcascades + cascade_name

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():

        print(f"Error: {cascade_name} 로드 실패")

        return None

    return cascade

# 폴더 생성 함수

def create_folders(paths):

    """필요한 폴더 생성"""

    for path in paths:

        if not os.path.exists(path):

            os.makedirs(path)

def apply_mosaic(frame, faces, block_size=15):
    """감지된 얼굴 영역에 모자이크를 적용한다."""
    h_img, w_img = frame.shape[:2]

    for (x, y, w, h) in faces:
        # 프레임 경계를 벗어나는 좌표를 보정한다.
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        # 얼굴 영역 추출
        face_roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = face_roi.shape[:2]

        # 축소 (다운샘플링)
        down_w = max(1, roi_w // block_size)
        down_h = max(1, roi_h // block_size)
        small = cv2.resize(face_roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)

        # 다시 확대 (업샘플링) - INTER_NEAREST: 블록화 효과
        mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        # 모자이크된 영역을 원본에 복사
        frame[y1:y2, x1:x2] = mosaic

    return frame

# Cascade 로드

face_cascade = cv2.CascadeClassifier(

    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

)

# 웹캠 시작

cap = cv2.VideoCapture(0)

print("웹캠 모자이크 처리 시작... (q로 종료)")

while True:

    ret, frame = cap.read()

    

    if not ret:

        break

    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    

    # 모자이크 적용

    frame = apply_mosaic(frame, faces, block_size=15)

    

    cv2.imshow('Face Mosaic', frame)

    

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()

cv2.destroyAllWindows()
