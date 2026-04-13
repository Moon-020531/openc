import os
import cv2
import numpy as np
import dlib


def find_predictor_path():
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "shape_predictor_68_face_landmarks.dat"),
        os.path.join(base, "..", "day10", "shape_predictor_68_face_landmarks.dat"),
    ]
    for path in candidates:
        full = os.path.abspath(path)
        if os.path.exists(full):
            return full
    raise FileNotFoundError("shape_predictor_68_face_landmarks.dat 파일을 찾을 수 없습니다.")


def shape_to_points(shape):
    pts = []
    for i in range(68):
        part = shape.part(i)
        pts.append((part.x, part.y))
    return pts


def get_triangles(img, points):
    h, w = img.shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for p in points:
        subdiv.insert(p)

    triangle_list = subdiv.getTriangleList()
    triangles = []

    for t in triangle_list:
        pt = t.reshape(-1, 2)
        if (pt < 0).any() or (pt[:, 0] >= w).any() or (pt[:, 1] >= h).any():
            continue

        idx = []
        for i in range(3):
            for j, p in enumerate(points):
                if abs(pt[i][0] - p[0]) < 1.0 and abs(pt[i][1] - p[1]) < 1.0:
                    idx.append(j)
                    break

        if len(idx) == 3:
            triangles.append(idx)

    return triangles


def warp_triangle(img1, img2, pts1, pts2):
    x1, y1, w1, h1 = cv2.boundingRect(np.float32([pts1]))
    x2, y2, w2, h2 = cv2.boundingRect(np.float32([pts2]))

    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return

    roi1 = img1[y1:y1 + h1, x1:x1 + w1]
    roi2 = img2[y2:y2 + h2, x2:x2 + w2]

    offset1 = np.zeros((3, 2), dtype=np.float32)
    offset2 = np.zeros((3, 2), dtype=np.float32)

    for i in range(3):
        offset1[i] = (pts1[i][0] - x1, pts1[i][1] - y1)
        offset2[i] = (pts2[i][0] - x2, pts2[i][1] - y2)

    mtrx = cv2.getAffineTransform(offset1, offset2)
    warped = cv2.warpAffine(
        roi1,
        mtrx,
        (w2, h2),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((h2, w2), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(offset2), 255)

    warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
    roi2_masked = cv2.bitwise_and(roi2, roi2, mask=cv2.bitwise_not(mask))
    img2[y2:y2 + h2, x2:x2 + w2] = roi2_masked + warped_masked


def swap_one_direction(src_img, dst_img, src_points, dst_points):
    draw = dst_img.copy()

    hull_index = cv2.convexHull(np.array(dst_points), returnPoints=False)
    hull_src = [src_points[int(i)] for i in hull_index.flatten()]
    hull_dst = [dst_points[int(i)] for i in hull_index.flatten()]

    triangles = get_triangles(dst_img, hull_dst)

    for tri in triangles:
        t1 = [hull_src[tri[j]] for j in range(3)]
        t2 = [hull_dst[tri[j]] for j in range(3)]
        warp_triangle(src_img, draw, t1, t2)

    mask = np.zeros(dst_img.shape, dtype=dst_img.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull_dst), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull_dst]))
    center = (r[0] + r[2] // 2, r[1] + r[3] // 2)

    return cv2.seamlessClone(np.uint8(draw), dst_img, mask, center, cv2.NORMAL_CLONE)


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(find_predictor_path())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("웹캠 얼굴 스왑 시작 (q: 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        output = frame.copy()

        if len(faces) >= 2:
            shape1 = predictor(gray, faces[0])
            shape2 = predictor(gray, faces[1])

            points1 = shape_to_points(shape1)
            points2 = shape_to_points(shape2)

            temp = swap_one_direction(frame, output, points1, points2)
            output = swap_one_direction(frame, temp, points2, points1)
        else:
            cv2.putText(
                output,
                "Need at least 2 faces",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Face Swap Webcam", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(e)