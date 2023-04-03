# author: young
import json
import cv2
import numpy as np

img_left_rectified = cv2.imread('./images/rect_left_image.png')
img_right_rectified = cv2.imread('./images/rect_right_image.png')

imgL = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('SGBM', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
# cv2.createTrackbar('preFilterCap', 'SGBM', 100, 1000, lambda x: None)
# cv2.createTrackbar('minDisparity', 'SGBM', 0, 100, lambda x: None)
# cv2.createTrackbar('num', 'SGBM', 155, 255, lambda x: None)
# cv2.createTrackbar('blockSize', 'SGBM', 1, 20, lambda x: None)
# cv2.createTrackbar('uniquenessRatio', 'SGBM', 32, 100, lambda x: None)
# cv2.createTrackbar('speckleWindowSize', 'SGBM', 225, 300, lambda x: None)
# cv2.createTrackbar('speckleRange', 'SGBM', 1, 20, lambda x: None)
# cv2.createTrackbar('disp12MaxDiff', 'SGBM', 1, 100, lambda x: None)
# cv2.createTrackbar('P1', 'SGBM', 3, 100, lambda x: None)
# cv2.createTrackbar('P2', 'SGBM', 5, 100, lambda x: None)

preFilterCap = 100
minDisparity = 0
numDisparities = 154
blockSize = 1
uniquenessRatio = 32
speckleWindowSize = 225
speckleRange = 1
disp12MaxDiff = 1
P1 = 3
P2 = 5
'''
# parameter
minDisparity：最小可能的差异值。正常情况下，它为零，但有时校正算法会移动图像，因此需要相应地调整此参数。
numDisparities：disparity的搜索范围，即最大差异减去最小差异。该值始终大于零。在当前的实现中，这个参数必须可以被16整除。
blockSize：块的线性大小。大小应该是奇数（因为块位于当前像素的中心）。更大的块大小意味着更平滑，但不太准确的视差图。较小的块大小会给出更详细的视差图，但算法找到错误对应的几率更高。一般在3到11之间。
P1：控制视差平滑度的第一个参数，是相邻像素之间视差变化为1的惩罚。值越大，视差越平滑。
P2：控制视差平滑度的第二个参数，是相邻像素之间视差变化超过1的惩罚。值越大，视差越平滑。该算法要求P2>P1。
disp12MaxDiff：左右视差检查中允许的最大差异（以整数像素为单位）。将其设置为非正值以禁用检查。
preFilterCap：预滤波图像像素的截断值。该算法首先计算每个像素的x方向的导数，并按[-preFilterCap，preFilterCap]间隔剪裁其值。结果值被传递到Birchfield-Tomasi像素代价函数。
uniquenessRatio：最佳（最小）计算成本函数值应超过第二最佳值的百分比，满足此百分比的条件下才认为找到的匹配是正确的。通常，5-15范围内的值就足够好了。
speckleWindowSize：视差连通区域像素点个数的大小。对于每一个视差点，当其连通区域的像素点个数小于speckleWindowSize时，认为该视差值无效，是噪点。把它设置为0以禁用斑点过滤。否则，将它设置在50-200范围内的某个地方。
speckleRange：视差连通条件，在计算一个视差点的连通区域时，当下一个像素点视差变化绝对值大于speckleRange就认为下一个视差像素点和当前视差像素点是不连通的。它将被隐式地乘以16。通常，1或2就足够了。
'''

# while 1:
# 两个trackbar用来调节不同的参数查看效果
# preFilterCap = cv2.getTrackbarPos('preFilterCap', 'SGBM')
# minDisparity = cv2.getTrackbarPos('minDisparity', 'SGBM')
# numDisparities = cv2.getTrackbarPos('num', 'SGBM')
# blockSize = cv2.getTrackbarPos('blockSize', 'SGBM')
# uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'SGBM')
# speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'SGBM')
# speckleRange = cv2.getTrackbarPos('speckleRange', 'SGBM')
# disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'SGBM')
# P1 = cv2.getTrackbarPos('P1', 'SGBM')
# P2 = cv2.getTrackbarPos('P2', 'SGBM')
# if numDisparities == 0:
#     numDisparities = 1
# numDisparities视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
# blockSize：SAD窗口大小，5~21之间为宜

stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                               numDisparities=numDisparities * 16,
                               blockSize=2 * blockSize + 1,
                               P1=8 * P1 ** 2,
                               P2=32 * P2 ** 2,
                               disp12MaxDiff=disp12MaxDiff,
                               preFilterCap=preFilterCap,
                               uniquenessRatio=uniquenessRatio,
                               speckleWindowSize=speckleWindowSize,
                               speckleRange=speckleRange)
# stereo = cv2.StereoSGBM_create(numDisparities=numDisparities * 16, blockSize=blockSize)
disparity = stereo.compute(imgL, imgR)

# 计算出的视差是CV_16S格式-16位有符号整数（-32768…32767）
disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)


disparity = cv2.applyColorMap(disparity, 2)
cv2.imshow('SGBM', disparity)
key = cv2.waitKey(0)
if key == ord("q"):
    # break
    cv2.destroyAllWindows()
