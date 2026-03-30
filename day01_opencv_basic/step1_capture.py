import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
   
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame', gray)
    key = cv.waitKey(1)
    
    if key == ord('q'):
        break
    
    elif key == ord('c'):
        cv.imwrite('my_photo.png', gray)

cap.release()
cv.destroyAllWindows()