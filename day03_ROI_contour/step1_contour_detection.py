import urllib.request
import os
import cv2 as cv
import numpy as np

filtered_count = 0  # 파란색으로 칠해진 조건에 맞는 도형들
excluded_count = 0  # 조건에 안 맞아서 걸러진 도형들 (너무 크거나 너무 작은 노이즈)

def download_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

# 사용할 샘플 이미지 다운로드
img = cv.imread(download_sample("pic1.png"), cv.IMREAD_GRAYSCALE)

_, binary= cv.threshold(img, 127, 255,cv.THRESH_BINARY_INV)

contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)



cv.drawContours(img_color, contours, -1, (0, 255, 0), 2)

for cnt in contours:
    area = cv.contourArea(cnt)
    if 100 < area < 5000:
        cv.drawContours(img_color, [cnt], 0, (255, 0, 0), 2)
        filtered_count += 1
    else:
        excluded_count += 1    
        
print(f"- 필터링된 컨투어 개수: {filtered_count}")
print(f"- 제외된 노이즈: {excluded_count}")
print(f"(참고: 화면에서 찾은 전체 윤곽선 개수는 {len(contours)}개 입니다.)")        
        
cv.imshow('Filtered Contours', img_color)
cv.waitKey(0)
cv.destroyAllWindows()        
