import cv2
import numpy as np

img = cv2.imread(r'D:\projects\opencv_programming\day03_ROI_contour\img\bad_rect.png')
img2 = img.copy()

# 그레이 스케일로 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 첫 번째 컨투어 가져오기
contour = contours[0]

# 전체 둘레의 0.05로 오차 범위 지정 
epsilon = 0.05 * cv2.arcLength(contour, True)

# 근사 컨투어 계산
approx = cv2.approxPolyDP(contour, epsilon, True)

# 컨투어 그리기
cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
cv2.drawContours(img2, [approx], -1, (0, 255, 0), 3)

# 결과 출력
cv2.imshow('contour', img)
cv2.imshow('approx', img2)
cv2.waitKey(0)  # 0을 명시적으로 넣어주는 것이 안전합니다.
cv2.destroyAllWindows()