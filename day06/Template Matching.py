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


# # 이미지 로드
# img = cv.imread(get_sample('messi5.jpg'))
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # 템플릿 추출: 사이트에서 파일 받기
# template = cv.imread('template.jpg', cv.IMREAD_GRAYSCALE)

# # Template Matching — TM_CCOEFF_NORMED 방법
# result = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)

# # 최적 매칭 위치 찾기
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
# top_left = max_loc  # TM_CCOEFF_NORMED에서는 max_loc이 최적

# # 매칭 영역 크기 계산
# h, w = template.shape[:2]
# bottom_right = (top_left[0] + w, top_left[1] + h)

# # 결과 표시
# result_img = img.copy()
# cv.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)

# # 화면에 표시
# cv.imshow('Template', cv.cvtColor(template, cv.COLOR_GRAY2BGR))
# cv.imshow('Result', result_img)
# cv.imshow('Result Map', result)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 결과 저장
# cv.imwrite('template_matching_result.jpg', result_img)


# # 이미지 로드
# img = cv.imread(get_sample('messi5.jpg'))
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# template = gray[80:230, 20:150]

# # 6가지 매칭 방법
# methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED',
#            'TM_SQDIFF', 'TM_SQDIFF_NORMED']

# results = []
# for method_name in methods:
#     # 메서드 타입 선택
#     method = getattr(cv, method_name)
    
#     # Template Matching 실행
#     result = cv.matchTemplate(gray, template, method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    
#     # TM_SQDIFF* 계열은 min_loc이 최적, 나머지는 max_loc
#     if 'SQDIFF' in method_name:
#         top_left = min_loc
#         score = min_val
#     else:
#         top_left = max_loc
#         score = max_val
    
#     results.append((method_name, score, top_left))
    
#     # 결과 출력
#     print(f"{method_name:15} → score={score:.4f}, top_left={top_left}")

# # 최고 점수 방법 표시
# best_method, best_score, best_loc = max(results, key=lambda x: x[1] if 'SQDIFF' not in x[0] else -x[1])
# print(f"\n최고 성능: {best_method} (score={best_score:.4f})")

# 이미지 로드
img = cv.imread(get_sample('messi5.jpg'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
template = gray[80:230, 20:150]

# Template Matching
result = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)

# 임계값 이상의 모든 매칭 위치 찾기
threshold = 0.8  # 80% 이상 유사도
locations = np.where(result >= threshold)

# 모든 후보 표시
result_img = img.copy()
h, w = template.shape[:2]

for y, x in zip(locations[0], locations[1]):
    top_left = (x, y)
    bottom_right = (x + w, y + h)
    cv.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 1)

cv.imshow('All Matches Above Threshold', result_img)
cv.waitKey(0)
cv.destroyAllWindows()

print(f"Found {len(locations[0])} matches above {threshold} threshold")
