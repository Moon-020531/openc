import cv2 as cv
import numpy as np


def nothing(x):
    pass

cv.namedWindow("color_detection")

cv.createTrackbar('h_min','color_detection',0,179,nothing)
cv.createTrackbar('s_min','color_detection',0,255,nothing)
cv.createTrackbar('v_min','color_detection',0,255,nothing)
cv.createTrackbar('h_max','color_detection',179,179,nothing)
cv.createTrackbar('s_max','color_detection',255,255,nothing)
cv.createTrackbar('v_max','color_detection',255,255,nothing)


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다")
    exit()

last_status = None
last_count = -1

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다")
        break

    frame_h, frame_w = frame.shape[:2]
    roi_x1 = int(frame_w * 0.2)
    roi_y1 = int(frame_h * 0.2)
    roi_x2 = int(frame_w * 0.8)
    roi_y2 = int(frame_h * 0.8)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    H_min = cv.getTrackbarPos('h_min','color_detection')
    S_min = cv.getTrackbarPos('s_min','color_detection')
    V_min = cv.getTrackbarPos('v_min','color_detection')
    H_max = cv.getTrackbarPos('h_max','color_detection')
    S_max = cv.getTrackbarPos('s_max','color_detection')
    V_max = cv.getTrackbarPos('v_max','color_detection')

    lower_yellow = np.array([H_min, S_min, V_min])
    upper_yellow = np.array([H_max, S_max, V_max])

    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
    contours, _ = cv.findContours(mask_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detected_count = 0

    for contour in contours:
        area = cv.contourArea(contour)
        if area > 500:
            detected_count += 1
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(
                frame,
                (x + roi_x1, y + roi_y1),
                (x + w + roi_x1, y + h + roi_y1),
                (0, 255, 0),
                2,
            )

    cv.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    cv.putText(frame, "ROI", (roi_x1, roi_y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    status = "DETECTED" if detected_count > 0 else "NOT DETECTED"
    status_color = (0, 255, 0) if detected_count > 0 else (0, 0, 255)

    cv.putText(frame, f"Status: {status}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv.putText(frame, f"Count: {detected_count}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if status != last_status or detected_count != last_count:
        print(f"[STATUS] {status} | count={detected_count}")
        last_status = status
        last_count = detected_count

    cv.imshow("Yellow", frame)
    cv.imshow("Mask", mask_cleaned)

    # q를 누르면 종료
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()