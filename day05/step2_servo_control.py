import cv2 as cv
import numpy as np
import serial
import time

# 아두이노 연결
ser = serial.Serial('COM7', 9600)

def send_command(ser, command):
    try:
        ser.write(command.encode())
        return True
    except Exception as e:
        print(f"❌ 아두이노 전송 실패: {e}")
        return False

def nothing(x):
    pass

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다")
    exit()

lower_hsv = np.array([
    0,      # ← H 최소값 (0-180) : 흰색은 H 영향이 거의 없음
    0,      # ← S 최소값 (0-255) : 흰색은 채도가 낮음
    180     # ← V 최소값 (0-255) : 밝은 영역만 통과
])

upper_hsv = np.array([
    180,    # ← H 최대값 (0-180)
    45,     # ← S 최대값 (0-255) : 채도가 높은 잡색 제거
    255     # ← V 최대값 (0-255)
])

# 이전 상태 변수 초기화
previous_state = None
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

    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
    contours, _ = cv.findContours(mask_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detected_count = 0

    for contour in contours:
        area = cv.contourArea(contour)
        if area > 3000:
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
    current_state = detected_count > 0

    if previous_state is not None and current_state != previous_state:
        command = "O" if current_state else "C"
        if send_command(ser, command):
            print(f"[ARDUINO] {command} 전송")

    previous_state = current_state

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

