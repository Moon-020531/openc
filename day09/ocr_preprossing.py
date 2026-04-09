import cv2
import numpy as np
from pathlib import Path

# 번호판 이미지 로드 (Day 4에서 추출한 이미지)
base_dir = Path(__file__).resolve().parent
img_path = base_dir / '01가1134.jpg'
img_data = np.fromfile(str(img_path), dtype=np.uint8)
img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

# 1단계: 그레이스케일
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Step 1: Grayscale', gray)

# 2단계: CLAHE로 대비 최대화
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast = clahe.apply(gray)
cv2.imshow('Step 2: CLAHE', contrast)

# 3단계: 적응형 임계 처리
binary = cv2.adaptiveThreshold(
    contrast, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
cv2.imshow('Step 3: Binary', binary)

# 4단계: 윤곽선 검출 및 필터링
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

filtered = np.zeros_like(binary)
for contour in contours:
    area = cv2.contourArea(contour)
    if 70 < area < 500:
        cv2.drawContours(filtered, [contour], 0, 255, -1)

cv2.imshow('Step 4: Filtered Contours', filtered)

# 5단계: 저장
cv2.imwrite(str(base_dir / 'preprocessed_plate.jpg'), binary)

cv2.waitKey(0)
cv2.destroyAllWindows()