# -*- coding: UTF-8 -*-
import os
import re

import cv2
import numpy as np

from StereoRectify import get_rectify
from tools.DebugTools import timeit


def get_coordinate(point_left, point_right, Q):
	d = abs(point_left[0] - point_right[0])
	b = 1 / np.array(Q)[3, 2]
	f = np.array(Q)[2, 3]
	Z = f * b / d
	X = point_left[0] * b / d
	Y = point_left[1] * b / d
	# print(d)
	return [X, Y, Z]


def is_gray(image):
	"""
	判断图像是否为灰度图
	"""
	return len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)


@timeit
def get_match(image: np.ndarray, template: np.ndarray, method_flag: int = 5, start_coordinate: tuple = (0, 0), end_coordinate: tuple = (0, 0)):
	"""
	Args:
		image: image of 3-D
		template: image of 3-D
		method_flag: index of [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
		start_coordinate: start point of match, default (0, 0)
		end_coordinate: end point of match, default (0, 0)

	Returns: aoi of match, coordinate of top_left, coordinate of bottom_right

	"""
	# 转换为灰度图像
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	height, width = template_gray.shape
	# 设置模板匹配的起始位置
	start_x = start_coordinate[0]
	start_y = start_coordinate[1]
	# 限制模板匹配的搜索范围
	roi = img_gray[start_y:end_coordinate[1], start_x:end_coordinate[0]]
	
	# 进行模板匹配
	methods = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
	result = cv2.matchTemplate(roi, template_gray, methods[method_flag])
	# 获取匹配结果中最大的相似度值和对应的位置
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
	
	if method_flag in [0, 1]:
		loc = min_loc
	else:
		loc = max_loc
	# 计算匹配区域在原始图像中的坐标
	top_left = (loc[0] + start_x, loc[1] + start_y)
	bottom_right = (loc[0] + width - 1 + start_x, loc[1] + height - 1 + start_y)  # 模板大小为(width, height)，为保证抠出区域和模板大小相同，所以右下角坐标需-1
	aoi = image[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]  # 切片时不包含最后一个值，所以要+1
	return aoi, top_left, bottom_right


def get_rect_center(top_left, bottom_right):
	x1, y1 = top_left
	x2, y2 = bottom_right
	return ((x1 + x2) / 2, (y1 + y2) / 2)


def init_plane(points, degree=1):
	import matplotlib.pyplot as plt
	from scipy import interpolate
	
	# # 计算平面拟合方程
	# A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
	# b = points[:, 2]
	# coeff, r, rank, s = np.linalg.lstsq(A, b, rcond=None)
	#
	# # 输出拟合平面方程
	# print("拟合平面方程为: z = {:.4f}x + {:.4f}y + {:.4f}".format(coeff[0], coeff[1], coeff[2]))
	# # 绘制平面
	# x, y = np.meshgrid(range(50, 80), range(0, 60))
	# z = coeff[0] * x + coeff[1] * y + coeff[2]
	# ax.plot_surface(x, y, z, alpha=0.5)
	#
	# plt.show()
	# # return coeff
	
	# 使用 bisplrep 拟合曲面方程
	tck = interpolate.bisplrep(points[:, 0], points[:, 1], points[:, 2], kx=degree, ky=degree)
	# 生成曲面网格点
	x_new, y_new = np.linspace(min(points[:, 0]), max(points[:, 0]), 100), np.linspace(min(points[:, 1]), max(points[:, 1]), 100)
	x_new, y_new = np.meshgrid(x_new, y_new)
	z_new = interpolate.bisplev(x_new[0, :], y_new[:, 0], tck)
	# 绘制三维散点图和拟合面
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(points[:, 0], points[:, 1], points[:, 2])
	ax.plot_surface(x_new, y_new, z_new.T, cmap='coolwarm')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('Bivariate Spline Surface')
	plt.show()
	return tck


def get_plane_point(X, Y, tck):
	from scipy import interpolate
	
	# 给定X,Y的值，计算Z
	Z1 = interpolate.bisplev(X, Y, tck)
	print(f"({X}, {Y}) 在曲面上的 Z1 值为 {Z1:.5f}")
	
	# # 计算平面拟合方程
	# A = np.c_[points[:,0], points[:,1], np.ones(points.shape[0])]
	# b = points[:,2]
	# coeff, r, rank, s = np.linalg.lstsq(A, b, rcond=None)
	# Z2 = coeff[0]*X + coeff[1]*Y + coeff[2]
	# print(f"({X}, {Y}) 在曲面上的 Z2 值为 {Z2:.5f}")
	
	return (X, Y, Z1)


@timeit
def get_coordinates(image_left: str, image_right: str, templates_dir: str, result_dir: str, calibration_json: str, rectify_json: str):
	"""
	Args:
		image_left:
		image_right:
		templates_dir:
		result_dir:
		calibration_json: calibration_json file to load
		rectify_json: rectify_json file to save

	Returns:

	"""
	img_left = cv2.imread(image_left)
	img_right = cv2.imread(image_right)
	rect_left_image, rect_right_image, Q = get_rectify(img_left, img_right, calibration_json, rectify_json)
	
	coordinates = []
	template_images = [(int(re.findall('\d+', f)[0]), cv2.imread(os.path.join(templates_dir, f), 1)) for f in os.listdir(templates_dir) if 'template' in f]
	template_images.sort()
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	for i, (num, template) in enumerate(template_images):
		# path = os.path.join(templates_dir, filenames_list[i])
		# template = cv2.imread(path)
		# 寻找匹配区域
		aoi_left, left_top_left, left_bottom_right = get_match(rect_left_image, template, 5, (1800, 0), (3840, 2748))
		aoi_right, right_top_left, right_bottom_right = get_match(rect_right_image, aoi_left, 5, end_coordinate=(2000, 2748))
		# 寻找匹配区域中心
		left_mid = get_rect_center(left_top_left, left_bottom_right)
		right_mid = get_rect_center(right_top_left, right_bottom_right)
		# 绘制匹配区域与中心
		cv2.rectangle(rect_left_image, left_top_left, left_bottom_right, (0, 0, 255), 1)
		cv2.circle(rect_left_image, (int(left_mid[0]), int(left_mid[1])), 2, (0, 0, 255), -1)
		cv2.rectangle(rect_right_image, right_top_left, right_bottom_right, (0, 0, 255), 1)
		cv2.circle(rect_right_image, (int(right_mid[0]), int(right_mid[1])), 2, (0, 0, 255), -1)
		result = np.concatenate((rect_left_image, rect_right_image), axis=1)
		# 匹配对应点获取空间坐标[X,Y,Z]
		coordinate = get_coordinate(left_mid, right_mid, Q)
		coordinates.append(coordinate)
		# print(coordinate)
		cv2.imwrite(os.path.join(result_dir, str(num) + ".png"), result)
	coordinates = np.asarray(coordinates)
	return coordinates


# init_plate_coordinates = get_coordinates(r"D:\Fenkx\Fenkx - General\Ubei\Stereo\init_palte\Image_1.bmp",
# 										 r"D:\Fenkx\Fenkx - General\Ubei\Stereo\init_palte\Image_2.bmp",
# 										 r'D:\Fenkx\Fenkx - General\Ubei\Stereo\init_palte\template',
# 										 r'D:\Fenkx\Fenkx - General\Ubei\Stereo\init_palte\result',
# 										 r'./config/calibration_parameters.json',
# 										 r'./config/rectify_parameters.json')
# tac = init_plane(init_plate_coordinates, 1)
# # print(tac)
object_coordinates = get_coordinates(r"D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_12_new_new\Image_33.bmp",
									 r"D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_12_new_new\Image_34.bmp",
									 r'D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_12_new_new\template2',
									 r'D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_12_new_new\result',
									 r'./config/calibration_parameters.json',
									 r'./config/rectify_parameters.json')
# points = {}
# init_points = {}

# for object_coordinate in object_coordinates:
# 	init_X, init_Y, init_Z = get_plane_point(object_coordinate[0], object_coordinate[1], tac)
# 	depth = init_Z - object_coordinate[2]
# 	points[f"{init_X, init_Y}"] = depth
# 	init_points[f"{init_X, init_Y}"] = init_Z

print(object_coordinates)
# print(points)
