import cv2 as cv

import numpy as np

import matplotlib.pyplot as plt

from sample_download import get_sample

# ========== Step 1: 이미지 로드 ==========

# 공식 샘플 이미지 또는 자신의 이미지 사용

img1 = cv.imread(get_sample('box.png', repo='opencv'), cv.IMREAD_GRAYSCALE)

img2 = cv.imread(get_sample('box_in_scene.png', repo='opencv'), cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:

    print("Error: 이미지를 찾을 수 없습니다.")

    exit()

print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")


# ========== Step 2: 특징점 검출기 초기화 ==========

# SIFT 또는 ORB 선택

# TODO: 아래 중 하나를 선택해서 코드 작성하세요

# 1) SIFT (느리지만 정확): sift = cv.SIFT_create()
sift = cv.SIFT_create()

# 2) ORB (빠르지만 덜 정확): sift = cv.ORB_create()

# ========== Step 3: 키포인트와 디스크립터 추출 ==========

# TODO: kp1, des1 = sift.detectAndCompute(img1, None)

#       kp2, des2 = sift.detectAndCompute(img2, None)

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 위 두 줄을 구현하세요

print(f"Keypoints found - img1: {len(kp1)}, img2: {len(kp2)}")

# ========== Step 4: FLANN 매칭기 설정 ==========

# SIFT는 float descriptor → KDTREE

# ORB는 binary descriptor → LSH

# TODO: detector_type을 판단하고 FLANN 파라미터 설정하세요

# SIFT 사용 시:

#   FLANN_INDEX_KDTREE = 1
FLANN_INDEX_KDTREE = 1

#   index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#   search_params = dict(checks=50)
search_params = dict(checks=50)

# ORB 사용 시:

#   FLANN_INDEX_LSH = 6

#   index_params = dict(algorithm=FLANN_INDEX_LSH, 

#                       table_number=12, key_size=20, multi_probe_level=2)

#   search_params = dict(checks=50)

# flann = cv.FlannBasedMatcher(index_params, search_params)

# ========== Step 5: knnMatch로 k=2 매칭 ==========

# TODO: matches = flann.knnMatch(des1, des2, k=2)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)  

print(f"Total matches: {len(matches)}")

# ========== Step 6: Lowe's 비율 테스트 ==========

good_matches = []

for match_pair in matches:

    if len(match_pair) == 2:

        m, n = match_pair
        
        if m.distance < 0.7 * n.distance:

            good_matches.append(m)
        # TODO: m.distance < 0.7 * n.distance 를 확인하고
        # 조건을 만족하면 good_matches에 추가하세요

print(f"Good matches after Lowe's ratio test: {len(good_matches)}")

# ========== Step 7: 시각화 ==========
def draw_matches(img1, kp1, img2, kp2, matches, title="Matches"):
    # 1. OpenCV의 내장 함수인 cv.drawMatches를 사용해 매칭 결과를 그립니다.
    # flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS는 매칭되지 않은 특징점은 그리지 않게 해줍니다.
    matched_img = cv.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # 2. OpenCV는 BGR 형식으로 색상을 처리하고 Matplotlib은 RGB로 처리하므로 색상 공간을 변환합니다.
    # (선 색상이 정상적으로 보이게 하기 위함)
    matched_img_rgb = cv.cvtColor(matched_img, cv.COLOR_BGR2RGB)
    
    # 3. Matplotlib을 이용해 화면에 출력합니다.
    plt.figure(figsize=(12, 6))
    plt.imshow(matched_img_rgb)
    plt.title(title)
    plt.axis('off') # 축 눈금 숨기기
    plt.show()


if len(good_matches) >= 10:
    draw_matches(img1, kp1, img2, kp2, good_matches,

                 title=f"Good Matches ({len(good_matches)})")

else:

    print("Not enough good matches!")


