# -*- coding: UTF-8 -*-

import numpy as np
import cv2
from StereoRectify import get_rectify
from Temlate_Match import get_coordinate
from tools.SerializeTools import config2json


def coord_trans(coord) -> np.ndarray:
	"""
	以标定板左上角为原点，建立以标定网格边为ox，oy，垂直于标定板为oz的坐标系UVW
	Args:
		coord: coordinate of points, [o,x,y]

	Returns: M, Transformation matrix, XYZ-->UVW, P_uvw = np.dot(M, P_xyz)

	"""
	ox = coord[1] - coord[0]
	oy = coord[2] - coord[0]
	# 求垂直于平面 oxy 的向量 oz
	oz = np.cross(ox, oy)
	# Normalize the vectors
	u1 = ox / np.linalg.norm(ox)
	u2 = oy / np.linalg.norm(oy)
	u3 = oz / np.linalg.norm(oz)
	# 坐标系 XYZ 到坐标系 UVW 的转换矩阵
	
	M2 = np.vstack((np.column_stack((u1, u2, u3, np.array([0, 0, 0]))), [0, 0, 0, 1]))
	# M1 = np.vstack((np.column_stack((np.eye(3), coord[0])), [0, 0, 0, 1]))
	# M = np.dot(M1, M2)
	M = M2
	return M


CHECKERBOARD = (11, 8)  # 棋盘格内角点数
CHESSBOARD_SIZE = 1.5  # 棋盘格大小，单位mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)


def get_oxy(img_l, img_r):
	"""
	Args:
		img_l:
		img_r:

	Returns:pts_l, pts_r

	"""
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
	
	o_l = corners2_l[0][0].astype(int)
	x_l = corners2_l[77][0].astype(int)
	y_l = corners2_l[10][0].astype(int)
	o_r = corners2_r[0][0].astype(int)
	x_r = corners2_r[77][0].astype(int)
	y_r = corners2_r[10][0].astype(int)
	# # 绘制线段
	# img_l = cv2.line(img_l, o_l, x_l, (0, 0, 255), 1)
	# img_l = cv2.line(img_l, o_l, y_l, (0, 0, 255), 1)
	# img_l = cv2.line(img_l, o_r, x_r, (0, 255, 0), 1)
	# img_l = cv2.line(img_l, o_r, y_r, (0, 255, 0), 1)
	# cv2.imwrite(r'D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_12_new_new\corner_l.png', img_l)
	# cv2.imwrite(r'D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_12_new_new\corner_r.png', img_r)
	
	# 定义坐标点
	pts_l = np.array([o_l, x_l, y_l], np.int32)
	pts_r = np.array([o_r, x_r, y_r], np.int32)
	return pts_l, pts_r


chessboard_left_path = r"D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_12_new_new\3.bmp"
chessboard_right_path = r"D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_12_new_new\4.bmp"
chessboard_left = cv2.imread(chessboard_left_path, 1)
chessboard_right = cv2.imread(chessboard_right_path, 1)
rect_left_image, rect_right_image, Q = get_rectify(chessboard_left, chessboard_right, r'./config/calibration_parameters.json', r'./config/rectify_parameters.json')

pts_l, pts_r = get_oxy(rect_left_image, rect_right_image)

coord = np.array([get_coordinate(pts_l[i], pts_r[i], Q) for i in range(3)])
T = coord_trans(coord)
print(T)
coord_trans_dict = {
	'T': T.tolist()
}
config2json(coord_trans_dict, './config/coord_trans_dict.json')
