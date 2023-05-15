# -*- coding: UTF-8 -*-

import numpy as np
import cv2
from StereoRectify import get_rectify_images
from Temlate_Match import get_coordinate
from tools.SerializeTools import config2json


def coord_trans(ox, oy) -> np.ndarray:
	"""
	# 以标定板左上角为原点，建立以标定网格边为ox，oy，垂直于标定板为oz的坐标系UVW
	:param ox: vector,coord(x)-coord(o)
	:param oy: vector,coord(y)-coord(o)
	:return: T, Transformation matrix, XYZ-->UVW, P_uvw = np.dot(T, P_xyz)
	"""
	# 求垂直于平面 oxy 的向量 oz
	oz = np.cross(ox, oy)
	
	# # Gram-Schmidt 正交化
	# u1 = ox / np.linalg.norm(ox)
	# p2 = np.dot(oy, u1) * u1
	# u2 = (oy - p2) / np.linalg.norm(oy - p2)
	# p3 = np.dot(oz, u1) * u1 + np.dot(oz, u2) * u2
	# u3 = (oz - p3) / np.linalg.norm(oz - p3)

	# Normalize the vectors
	u1 = ox / np.linalg.norm(ox)
	u2 = oy / np.linalg.norm(oy)
	u3 = oz / np.linalg.norm(oz)
	# 坐标系 XYZ 到坐标系 UVW 的转换矩阵
	T = np.vstack((u1, u2, u3)).T
	return T


CHECKERBOARD = (11, 8)  # 棋盘格内角点数
CHESSBOARD_SIZE = 1.5  # 棋盘格大小，单位mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)


def get_oxy(img_l, img_r):
	# img_l = cv2.imread(img_l_path, 1)
	gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
	# img_r = cv2.imread(img_r_path, 1)
	gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
	print(f"读取完成，寻找图片角点中...")
	ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD)  # 检测棋盘格内角点
	ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD)
	if ret_l and ret_r:
		# 提高角点精度
		corners2_l = cv2.cornerSubPix(gray_l, corners_l, (5, 5), (-1, -1), criteria)
		corners2_r = cv2.cornerSubPix(gray_r, corners_r, (5, 5), (-1, -1), criteria)
	o_l = corners2_l[0][0]
	x_l = corners2_l[77][0]
	y_l = corners2_l[10][0]
	o_r = corners2_r[0][0]
	x_r = corners2_r[77][0]
	y_r = corners2_r[10][0]
	# 定义坐标点
	pts_l = np.array([o_l, x_l, y_l], np.float32)
	pts_r = np.array([o_r, x_r, y_r], np.float32)
	return pts_l, pts_r


# chessboard_left_path = r"D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_edge\left\Image_243.bmp"
# chessboard_right_path = r"D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_edge\right\Image_244.bmp"
# chessboard_left = cv2.imread(chessboard_left_path, 1)
# chessboard_right = cv2.imread(chessboard_right_path, 1)
# rect_left_image, rect_right_image, Q = get_rectify_images(chessboard_left, chessboard_right, r'./config/calibration_parameters.json', r'./config/rectify_parameters.json')
#
# pts_l, pts_r = get_oxy(rect_left_image, rect_right_image)
# print(pts_l)
# coord = []
# for i in range(3):
# 	p1 = pts_l[i]
# 	p2 = pts_r[i]
# 	coord.append(get_coordinate(p1, p2, Q))
# coord = np.array(coord)
# ox = coord[1] - coord[0]
# oy = coord[2] - coord[0]
# T = coord_trans(ox, oy)
# print(T)
# coord_trans_dict = {
# 	'T': T.tolist()
# }
# config2json(coord_trans_dict, './config/coord_trans_dict.json')
