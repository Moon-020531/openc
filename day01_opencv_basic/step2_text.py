import cv2 as cv
import numpy as np

img = cv.imread('my_photo.png')
h, w = img.shape[:2]

overlay = img.copy()
cv.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
cv.addWeighted(overlay, 0.5, img, 0.5, 0, overlay)

cv.putText(overlay, 'Moonkyuseok', (10, h - 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) 
cv.putText(overlay, 'YH', (w - 130 , h - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv.imshow('Overlay Result', overlay)
cv.waitKey(0)
cv.destroyAllWindows()