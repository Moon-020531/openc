import cv2 as cv
import numpy as np
import urllib.request
import os

def nothing(x):
    pass

def get_sample(filename, flags=cv.IMREAD_COLOR):
    if not os.path.exists(filename):
        url=f"https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{filename}"
        urllib.request.urlretrieve(url, filename)
    return cv.imread(filename, flags)    

cv.namedWindow('image')


cv.createTrackbar('threshold', 'image', 127, 255, nothing)
cv.createTrackbar('blockSize', 'image', 11, 31, nothing)
cv.createTrackbar('C', 'image', 2, 20, nothing)

img = get_sample("sudoku.png", cv.IMREAD_GRAYSCALE)
assert img is not None, "Could not read the image."
img = cv.medianBlur(img, 5)

while(1):
    threshold = cv.getTrackbarPos('threshold', 'image')
    blockSize = cv.getTrackbarPos('blockSize', 'image')
    C = cv.getTrackbarPos('C', 'image')

    
    if blockSize < 3:
        blockSize = 3
    elif blockSize % 2 == 0:
        blockSize += 1

    # 단일 임계값 이진화
    ret, th1 = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    
    # Otsu의 이진화
    ret_otsu, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    print(f"Otsu가 결정한 임계값: {ret_otsu}")
   
    # 적응형 이진화 
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY, blockSize, C)    
    
    top_row = np.hstack((img, th1))      
    bottom_row = np.hstack((th2, th3))   
    combined = np.vstack((top_row, bottom_row))
    

    resized = cv.resize(combined, (400, 400))
    cv.imshow('image', resized)
   
    k = cv.waitKey(1) & 0xFF
    if k == 113:
        break


cv.destroyAllWindows()