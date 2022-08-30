import cv2
import numpy as np

class Detect(object):

    # 将RGB图像转为HSI图像 返回h通道，s通道，i通道图像以及hsi图像
    def rgb2hsi(self, image):
        b, g, r = cv2.split(image)  # 读取通道
        r = r / 255.0  # 归一化
        g = g / 255.0
        b = b / 255.0
        eps = 1e-6  # 防止除零

        img_i = (r + g + b) / 3  # I分量

        img_h = np.zeros(r.shape, dtype=np.float32)
        img_s = np.zeros(r.shape, dtype=np.float32)
        min_rgb = np.zeros(r.shape, dtype=np.float32)
        # 获取RGB中最小值
        min_rgb = np.where((r <= g) & (r <= b), r, min_rgb)
        min_rgb = np.where((g <= r) & (g <= b), g, min_rgb)
        min_rgb = np.where((b <= g) & (b <= r), b, min_rgb)
        img_s = 1 - 3 * min_rgb / (r + g + b + eps)  # S分量

        num = ((r - g) + (r - b)) / 2
        den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
        theta = np.arccos(num / (den + eps))
        img_h = np.where((b - g) > 0, 2 * np.pi - theta, theta)  # H分量
        img_h = np.where(img_s == 0, 0, img_h)

        img_h = img_h / (2 * np.pi)  # 归一化
        temp_s = img_s - np.min(img_s)
        temp_i = img_i - np.min(img_i)
        img_s = temp_s / np.max(temp_s)
        img_i = temp_i / np.max(temp_i)

        image_hsi = cv2.merge([img_h, img_s, img_i])
        return img_h, img_s, img_i, image_hsi

    def nothing(self, x):
        pass

    # 图像预处理与轮廓获取
    def findcont(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        reval_T_2, dst_Tri_2 = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)
        dst_Tri_2 = ~dst_Tri_2  # 黑白像素转置
        contours, hierarchy = cv2.findContours(dst_Tri_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy

    # 获取最大面积轮廓的凸包并返回
    def get_hull(self, contours):
        # 找到最大面积的轮廓
        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
        max_index = np.argmax(np.array(area))
        # 获取最大面积的凸包[hull]
        hull = cv2.convexHull(contours[max_index])
        return hull

    # 获取凸包hull的roi区域并返回
    def return_roi(self, height, width, hull):
        roi = np.zeros((height, width), dtype=np.uint8)
        roi.fill(128)
        cv2.fillPoly(roi, [hull], (255), 8, 0)
        return roi

    # 获取图像中黑色像素的个数
    def get_blackNum(self, height, width, image):
        # 建立一个与图像尺寸相同的全零数组
        npim = np.zeros((height, width), dtype=np.uint8)
        # 将图像3个通道相加赋值给空数组
        npim[:] = image[:, :]
        cont = len(npim[npim == 0])
        return cont

    # 判断是否是好果子
    def judge_ifgood(self, cont, hull_area, image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if (cont / hull_area <= 0.2):
            cv2.putText(image, "This is good", (0, 250), font, 3, (0, 255, 0), 15,
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
        else:
            cv2.putText(image, "This is bad", (0, 250), font, 3, (0, 255, 0), 15,
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
        return image