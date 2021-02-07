import numpy as np
import cv2


def video_capture():
    cap = cv2.VideoCapture(0)
    while True:
        # capture frame-by-frame
        ret, frame_origin = cap.read()

        # our operation on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 可选择灰度化

        # display the resulting frame
        # frame = hsv_detect(frame)

        frame = cv2.resize(frame_origin, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_CUBIC)  # 比例因子：fx=0.5,fy=0.5

        frame = ellipse_detect(frame)
        frame = img_blur(frame)
        frame = canny_detect(frame)
        # frame = dilate_demo(frame)

        cv2.imshow('frame_origin', frame_origin)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            break
    # when everything done , release the capture
    cap.release()
    cv2.destroyAllWindows()


def ellipse_detect(img):
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 15),
                43, 0, 360, (255, 255, 255), -1)

    YCRCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(YCRCB)
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            CR = YCRCB[i, j, 1]
            CB = YCRCB[i, j, 2]
            if skinCrCbHist[CR, CB] > 0:
                skin[i, j] = 255
    dst = cv2.bitwise_and(img, img, mask=skin)
    return dst


def hsv_detect(image):
    """
    :param image: 图片路径
    :return: None
    """
    img = image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (_h, _s, _v) = cv2.split(hsv)
    skin = np.zeros(_h.shape, dtype=np.uint8)
    (x, y) = _h.shape

    for i in range(0, x):
        for j in range(0, y):
            if (_h[i][j] > 7) and (_h[i][j] < 20) and (_s[i][j] > 28) and (_s[i][j] < 255) and (_v[i][j] > 50) and (
                    _v[i][j] < 255):
                skin[i][j] = 255
        else:
            skin[i][j] = 0

    dst = cv2.bitwise_and(img, img, mask=skin)
    return dst


# 膨胀
def dilate_demo(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 定义结构元素的形状和大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 膨胀操作
    dst = cv2.dilate(binary, kernel)
    return dst


# 腐蚀
def erode_demo(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 定义结构元素的形状和大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # 腐蚀操作
    dst = cv2.erode(binary, kernel)
    return dst


# 滤波
def img_blur(image):
    # 腐蚀操作
    # img_erode = erode_demo(image)
    # 膨胀操作
    img_dilate = dilate_demo(image)

    # 均值滤波
    # blur = cv2.blur(image, (5, 5))
    # 高斯滤波
    blur = cv2.GaussianBlur(img_dilate, (3, 3), 0)
    return blur


# Canny边缘检测v
def canny_detect(image):
    edges = cv2.Canny(image, 50, 200)
    return edges

    # 轮廓匹配


video_capture()
