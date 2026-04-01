import urllib.request
import os
import cv2 as cv
import numpy as np

def download_sample(filename):
    if not os.path.exists(filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return filename

# 사용할 샘플 이미지 다운로드
img = cv.imread(download_sample("pic1.png"), cv.IMREAD_GRAYSCALE)

_, binary= cv.threshold(img, 127, 255,cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

cv.drawContours(img_color, contours, -1, (0, 255, 0), 2)

for cnt in contours:
    area = cv.contourArea(cnt)
    if 100 < area < 10000:
        cv.drawContours(img_color, [cnt], 0, (255, 0, 0), 2)
        
cv.imshow('Filtered Contours', img_color)
cv.waitKey(0)
cv.destroyAllWindows()        
