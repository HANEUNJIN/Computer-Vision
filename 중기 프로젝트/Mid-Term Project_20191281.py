import numpy as np, cv2, math
def draw_histo(hist, shape=(200, 256, 3)):
    hsv_palate = make_palette(hist.shape[0])
    hist_img = np.full(shape, 255, np.uint8)
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX) # 정규화
    gap = hist_img.shape[1]/hist.shape[0] # 한 계급 너비

    for i, h in enumerate(hist):
        x = int(round(i*gap)) # 막대 사각형 시작 x 좌표
        w = int(round(gap)) # 그래프 너비 / 계급 개수 -> 한 계급 너비

        color = tuple(map(int, hsv_palate[i][0]))
        cv2.rectangle(hist_img, (x, 0, w, int(h)), color, cv2.FILLED)
        # cv2.rectangle(hist_img, (x, 0, w, int(h)), 0, cv2.FILLED) # 50: 회색
    return cv2.flip(hist_img, 0) # 영상 상하 뒤집기 후 반환
def make_palette(rows):
    hue = [round(i*180/rows)for i in range(rows)]
    hsv = [[(h,255,255)]for h in hue]
    hsv = np.array(hsv, np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

Mid_Term_Image = cv2.imread('Mid-Term_Image.jpg', cv2.IMREAD_COLOR) # 컬러 영상 읽기
if Mid_Term_Image is None: raise Exception("error") # 예외처리

HSV_img = cv2.cvtColor(Mid_Term_Image, cv2.COLOR_BGR2HSV) # OpenCV 함수
# hist = cv2.calcHist([HSV_img], [0], None, [32], [0, 256])
hist = cv2.calcHist([HSV_img], [0], None, [18], [0, 180])
hist_img = draw_histo(hist, (200, 360, 3))

cv2.imshow("Mid_Term_Image", Mid_Term_Image) # 화면에 이미지 보이지
cv2.imshow("Mid_Term_histogram_img", hist_img) # 화면에 이미지 보이기
cv2.waitKey(0) # 무한대기