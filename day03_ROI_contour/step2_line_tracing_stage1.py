import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("웰캠을 열 수 없습니다.")
    exit()
    
cv.namedWindow("Line Tracing stage 1", cv.WINDOW_NORMAL)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break 
    
    # 그레이스케일 변환
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # 이진화(Otsu)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    #컨투어 검출
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    #가장큰 컨투어 찾기
    largest_cnt=None
    max_area=0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_cnt = cnt
            
    #중심좌표 계산(최소 크기 필터)
    if largest_cnt is not None and max_area > 100:
        M = cv.moments(largest_cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # 컨투어 그리기
            cv.drawContours(frame, [largest_cnt], 0, (0, 255, 0), 2)
            
            # 중심점 그리기(빨강)
            cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            
            
            # 좌표 표시
            cv.putText(frame,f'Center: ({cX}, {cY})', (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        #이진화 이미지 + 원본 나란히 표시
        binary_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
        result = np.hstack((binary_color, frame))
        cv.imshow("Line Tracing stage 1", result)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()           