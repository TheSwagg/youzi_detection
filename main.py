from header import *

# 定义类
detect = Detect()
# 读取图像
img = cv2.imread("./youzi-picture/Image8.jpg")
img_2 = img.copy()
final_result = img.copy()
height, width, _ = img.shape
# 将RGB图像转为HSI图像，并且返回S通道图像
img_h, img_s, img_i, hsi_img = detect.rgb2hsi(img)
cv2.namedWindow("dst_Tri")
min_thre = 150
cv2.createTrackbar('thre', "dst_Tri", min_thre, 255, detect.nothing)           # createTrackbar

# while True:
cv2.imshow("img", img)      # 原图像
cv2.imshow("hsi_img", img_s)        # s通道图像

# k = cv2.waitKey(1) & 0xFF
# if k == 27:
#     break
thre = cv2.getTrackbarPos('thre', "dst_Tri")
reval_T, dst_Tri = cv2.threshold(img_s * 255, thre, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("dst_Tri", dst_Tri)      # 该图像为s通道阈值处理后的图像
draw_img = dst_Tri.copy()

# 图像预处理以及查找轮廓并返回
contours, hierarchy = detect.findcont(img_2)
# 获取凸包[hull]
hull = detect.get_hull(contours)
# 获取[hull]的roi区域
roi = detect.return_roi(height, width, hull)
cv2.imshow("roi", roi)      # roi区域图像
draw_img = cv2.bitwise_and(draw_img, draw_img, mask=roi)
cv2.imshow("draw_img", draw_img)        # roi区域提取后的图像

# 计算凸包面积
hull_area = cv2.contourArea(hull)
print("图像的宽度：", width)
print("图像的高度：", height)
print("图像的整体面积：", width * height)
print("获取凸包面积为：", hull_area)

# 遍历图像获取图像中黑色像素的个数
cont = detect.get_blackNum(height, width, draw_img)
print("图像中黑色像素的个数为：", cont)

# 最后进行判断，确定这个果子是否是一个好果子
# 如果黑色像素占整个凸包的白分子20，那么则选定这个果子是一个怀果子
final_result = detect.judge_ifgood(cont, hull_area, final_result)

cv2.imshow("final_result", final_result)

cv2.waitKey(0)