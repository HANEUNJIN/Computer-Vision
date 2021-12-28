import numpy as np, cv2, math
from Common.hough import accumulate, masking, select_lines # 허프 변환 함수 임포트

# 허프 변환 함수
def houghLines(src, rho, theta, thresh):
    acc_mat = accumulate(src, rho, theta)  # 허프 누적 행렬 계산
    acc_dst = masking(acc_mat, 7, 3, thresh)  # 마스킹 처리 7행,3열
    lines = select_lines(acc_dst, rho, theta, thresh)  # 임계 직선 선택
    return lines

# 검출 직선 그리기 함수
def draw_houghLines(src, lines, nline):
    if len(src.shape) < 3:
        dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)  # 컬러 영상 변환
    else:
        dst = np.copy(src)
    min_length = min(len(lines), nline)

    for i in range(min_length):
        rho, radian = lines[i, 0, 0:2]  # 수직거리, 각도 - 3차원 행렬임
        a, b = math.cos(radian), math.sin(radian)
        pt = (a * rho, b * rho)  # 검출 직선상의 한 좌표 계산
        delta = (-1000 * b, 1000 * a)  # 직선상의 이동 위치
        pt1 = np.add(pt, delta).astype('int')
        pt2 = np.subtract(pt, delta).astype('int')
        cv2.line(dst, tuple(pt1), tuple(pt2), (0, 255, 0), 2, cv2.LINE_AA) #녹색 직선 표시, 직선 두께 2

    return dst

# image = cv2.imread('Final-Term_Image.jpg', cv2.IMREAD_GRAYSCALE) # 흑백 이미지 불러오기
image = cv2.imread('Final-Term_Image.jpg', cv2.IMREAD_COLOR) # 컬러 이미지 불러오기
if image is None: raise Exception("영상 파일 읽기 에러")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 일단 그레이로 저장
blur = cv2.GaussianBlur(gray, (5, 5), 2, 2) # 가우시안 블러링
canny = cv2.Canny(blur, 50, 150, 5) # 캐니 에지 검출

rho, theta = 1,  np.pi / 180 # 수직거리 간격, 각도 간격(pi.180라디안)
lines1 = houghLines(canny, rho, theta, 80) # 저자 구현 함수
lines2 = cv2.HoughLines(canny, rho, theta, 80) # 허프라인 생성, OpenCV 함수

# my_dst = draw_houghLines(canny, lines2, 10) # 캐니에지에 허프라인
my_dst = draw_houghLines(image, lines2, 10) # 캐니에지에 허프라인
# dst1 = draw_houghLines(canny, lines1, 10)  # 직선 그리기
dst1 = draw_houghLines(image, lines1, 10)
# dst2 = draw_houghLines(canny, lines2, 10)
dst2 = draw_houghLines(image, lines2, 10)

# 이미지 보이기
cv2.imshow("image", image) # 컬러 이미지 보이기
# cv2.imshow("canny", canny)
cv2.imshow("my_dst", my_dst)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)