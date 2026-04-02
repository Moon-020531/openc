import cv2 as cv
import numpy as np

def find_license_plate(img):
    """
    자동차 번호판을 찾는 함수
    """
    
    height, width = img.shape[:2]
    
    # 1. 전처리
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 히스토그램 평균화(명암 개선)
    gray = cv.equalizeHist(gray)
    
    # 2. 에지 검출 + 모폴리지
    edges = cv.Canny(gray, 50, 150)
    
    # 가로선 강조(번호판은 가로 직사각형)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 5))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    # 3. 컨투어 검출
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # 번호판 기능 컨투어 필터링
    plate_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:  # 최소 면적 
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = w / h # 가로세로 비율
            
            #번호판: 가로가 세로의 3~6배
            if 3 < aspect_ratio < 6:
                plate_contours.append((x, y, w, h,area))
    
    # 4. 가장 큰 번호판 영역 선택 + 원근 반환
    if plate_contours:
        plate_contours.sort(key=lambda x: x[4], reverse=True)  # 면적 기준 정렬
        x, y, w, h, _ = plate_contours[0]
        
        # 원근 변환으로 정면 시점으로 정렬
        pts=np.float32([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
        dst_pts=np.float32([[0,0],[w,0],[0,h],[w,h]])
        M = cv.getPerspectiveTransform(pts, dst_pts)
        plate = cv.warpPerspective(img, M, (w, h))
        
        return plate, (x, y, w, h)
    
    return None, None


img = cv.imread('D:\projects\opencv_programming\day04\car_image.jpg')

if img is None:
    print("❌ 이미지를 불러올 수 없습니다. 'car_image.jpg' 파일을 확인하세요.")
    print("   또는 다음 명령으로 샘플 이미지를 다운로드하세요:")
    print("   img = cv.imread(get_sample('car_plate.jpg', repo='insightbook'))")
    exit()
    
# 번호판 추출
plate, rect = find_license_plate(img)

if plate is not None:
    x, y, w, h = rect
    
    # 원본에 검출 영역 표시
    result =img.copy()
    cv.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv.putText(result, "License Plate", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 원본 + 추출된 번호판 표시
    plate_resized = cv.resize(plate, (200, 100))
    result_resized = cv.resize(result, (640, 480))
    
    cv.imshow("Original with Detection", result_resized)
    cv.imshow("Extracted Plate", plate_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # 번호판 이미지에서 글자 찾기
plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
_, plate_binary = cv.threshold(plate_gray, 127, 255, cv.THRESH_BINARY)

# 글자 영역 컨투어 검출
contours, _ = cv.findContours(plate_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 각 글자의 bounding box 표시
for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    if w > 5 and h > 10:  # 글자 크기 필터
        cv.rectangle(plate, (x, y), (x+w, y+h), (0, 255, 0), 2) 
    
    # 추출 결과 저장
    cv.imwrite('license_plate_extracted.png', plate)
    print("✅ 번호판 추출 완료: license_plate_extracted.png")
else:
    print("❌ 번호판을 찾을 수 없습니다.")

    