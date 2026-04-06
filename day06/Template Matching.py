import cv2 as cv
import numpy as np
import urllib.request
import os
import matplotlib.pyplot as plt


def get_sample(filename):
    """OpenCV 공식 샘플 이미지 자동 다운로드"""
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename


# 이미지 로드
img = cv.imread(get_sample('messi5.jpg'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 템플릿 추출: 사이트에서 파일 받기
template = cv.imread('template.jpg', cv.IMREAD_GRAYSCALE)

# Template Matching — TM_CCOEFF_NORMED 방법
result = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)

# 최적 매칭 위치 찾기
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
top_left = max_loc  # TM_CCOEFF_NORMED에서는 max_loc이 최적

# 매칭 영역 크기 계산
h, w = template.shape[:2]
bottom_right = (top_left[0] + w, top_left[1] + h)

# 결과 표시
result_img = img.copy()
cv.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)

# 화면에 표시
cv.imshow('Template', cv.cvtColor(template, cv.COLOR_GRAY2BGR))
cv.imshow('Result', result_img)
cv.imshow('Result Map', result)
cv.waitKey(0)
cv.destroyAllWindows()

# 결과 저장
cv.imwrite('template_matching_result.jpg', result_img)
