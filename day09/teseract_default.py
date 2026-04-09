import cv2
import pytesseract
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 이미지 로드
image_path = base_dir / '01가1134.jpg'
image_data = np.fromfile(str(image_path), dtype=np.uint8)
img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
if img is None:
	raise FileNotFoundError(f'이미지를 읽을 수 없습니다: {image_path}')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 기본 OCR
text = pytesseract.image_to_string(gray)
print(f"인식 결과: {text}")

# 상세 정보 (신뢰도 포함)
data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
print(f"신뢰도: {data['conf']}")  # 각 글자의 신뢰도