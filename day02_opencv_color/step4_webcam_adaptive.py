import cv2 as cv
import numpy as np

def nothing(x):
    pass

cap = cv.VideoCapture(0)

cv.namedWindow('image')
cv.createTrackbar('blockSize', 'image', 11, 31, nothing)
cv.createTrackbar('C', 'image', 2, 20, nothing)

while(1):
    _, frame = cap.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    blockSize = cv.getTrackbarPos('blockSize', 'image')
    C = cv.getTrackbarPos('C', 'image')
    
    if blockSize < 3:
        blockSize = 3
    elif blockSize % 2 == 0:
        blockSize += 1
    
    Gaussian = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv.THRESH_BINARY, blockSize, C)
    
    Mean = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
                                cv.THRESH_BINARY, blockSize, C)
    
    ret_otsu, Otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    ret_global, Global = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
       
    
    cv.imshow('image', frame) 
    cv.imshow('Gaussian', Gaussian)
    cv.imshow('Otsu', Otsu)
    cv.imshow('Global', Global)
    cv.imshow('Mean', Mean)

   
    k = cv.waitKey(5) & 0xFF
    if k == 113:
        break


cap.release()
cv.destroyAllWindows()