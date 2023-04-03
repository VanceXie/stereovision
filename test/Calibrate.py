# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# 获取图像
img_left = cv2.imread('left.jpg', 0)
img_right = cv2.imread('right.jpg', 0)

# 创建SIFT特征点检测器
sift = cv2.xfeatures2d.SIFT_create()

# 提取特征点和描述符
kp1, des1 = sift.detectAndCompute(img_left, None)
kp2, des2 = sift.detectAndCompute(img_right, None)

# 创建FLANN特征点匹配器
flann = cv2.FlannBasedMatcher()

# 匹配特征点
matches = flann.knnMatch(des1, des2, k=2)

# 筛选出最优匹配点对
good_matches = []
for m, n in matches:
	if m.distance < 0.9 * n.distance:
		good_matches.append(m)

# 提取最优匹配点对的坐标
pts_left = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts_right = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 计算基础矩阵F
F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)

# 筛选出有效的匹配点对
pts_left = pts_left[mask.ravel() == 1]
pts_right = pts_right[mask.ravel() == 1]
# 定义相机内参矩阵K
K = np.array([[1000, 0, 500],
			  [0, 1000, 500],
			  [0, 0, 1]])

# 计算本质矩阵E
E = np.dot(np.dot(K.T, F), K)

# 计算相机矩阵P
points, R, t, mask = cv2.recoverPose(E, pts_left, pts_right, K)
P = np.hstack((R, t))

# 保存相机参数
np.savetxt('camera_matrix.txt', K)
np.savetxt('camera_pose.txt', P)
