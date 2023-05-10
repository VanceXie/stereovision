import os
import re

import cv2
import numpy as np

from tools.SerializeTools import config2json

img_path = r'D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_edge'

CHECKERBOARD = (11, 8)  # 棋盘格内角点数
CHESSBOARD_SIZE = 1.5  # 棋盘格大小，单位mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)

img_points_l = []  # 存放左图像坐标系下角点位置
img_points_r = []  # 存放右图像坐标系下角点位置
obj_points = []  # 存放世界坐标系下角点位置

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp[0, :, :2] *= CHESSBOARD_SIZE

# 将读取文件路径进行排序
filenames_l = os.listdir(os.path.join(img_path, 'left'))
filenames_r = os.listdir(os.path.join(img_path, 'right'))
# 找出字符串中的数字并依据其整形进行排序
# re.findall('\d+', x) 会返回一个列表，其中包含所有匹配的连续数字字符串
filenames_l.sort(key=lambda x: int(re.findall('\d+', x)[0]))
filenames_r.sort(key=lambda x: int(re.findall('\d+', x)[0]))
print(filenames_l)
print(filenames_r)

# 放置绘制角点图的路径
left_corner_path = os.path.join(img_path, 'left_corner')
right_corner_path = os.path.join(img_path, 'right_corner')
if not os.path.exists(left_corner_path):
	os.makedirs(left_corner_path)
if not os.path.exists(right_corner_path):
	os.makedirs(right_corner_path)
# 获取每张图棋盘格角点，将坐标添入矩阵
for index, (filename_l, filename_r) in enumerate(zip(filenames_l, filenames_r)):
	print(f"读取第{index + 1}图片中...")
	img_l = cv2.imread(os.path.join(img_path, 'left', filename_l))
	gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
	img_r = cv2.imread(os.path.join(img_path, 'right', filename_r))
	gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
	print(f"读取完成，寻找第{index + 1}图片角点中...")
	ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD)  # 检测棋盘格内角点
	ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD)
	if ret_l and ret_r:
		obj_points.append(objp)
		# 提高角点精度
		corners2_l = cv2.cornerSubPix(gray_l, corners_l, (5, 5), (-1, -1), criteria)
		img_points_l.append(corners2_l)
		corners2_r = cv2.cornerSubPix(gray_r, corners_r, (5, 5), (-1, -1), criteria)
		img_points_r.append(corners2_r)
		print(f"第{index + 1}图片角点寻找完成并加入列表，绘制角点中...")
		# 绘制角点
		img_l = cv2.drawChessboardCorners(img_l, CHECKERBOARD, corners2_l, ret_l)
		img_r = cv2.drawChessboardCorners(img_r, CHECKERBOARD, corners2_r, ret_r)
		
		index_name = '{:0>2d}'.format(index + 1)
		
		cv2.imwrite(os.path.join(left_corner_path, f'{index_name}_corner.bmp'), img_l)
		cv2.imwrite(os.path.join(right_corner_path, f'{index_name}_corner.bmp'), img_r)
		print(f"第{index + 1}图片角点绘制完成")
	else:
		print(f"第{index + 1}图片角点寻找失败，请检查图像")
# 先分别做单目标定
ret1, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(obj_points, img_points_l, gray_l.shape[::-1], None, None)
ret2, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(obj_points, img_points_r, gray_r.shape[::-1], None, None)
# 再做双目标定
retval, cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, R, T, E, F = cv2.stereoCalibrate(
		obj_points,
		img_points_l,
		img_points_r, mtx_l,
		dist_l, mtx_r,
		dist_r,
		gray_l.shape[::-1],
		flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_FIX_K3,
		criteria=criteria
)

# 将返回参数（相机内外参数）加入字典，并写入json配置文件
calibration_parameters_dict = {
	'cameraMatrix_left' : cameraMatrix_left.tolist(),
	'distCoeffs_left'   : distCoeffs_left.tolist(),
	'cameraMatrix_right': cameraMatrix_right.tolist(),
	'distCoeffs_right'  : distCoeffs_right.tolist(),
	'R'                 : R.tolist(),
	'T'                 : T.tolist(),
	'E'                 : E.tolist(),
	'F'                 : F.tolist()
}
config2json(calibration_parameters_dict, './config/calibration_parameters.json')
