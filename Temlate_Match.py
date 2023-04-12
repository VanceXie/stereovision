# -*- coding: UTF-8 -*-
import json

import cv2
import numpy as np

import StereoRectify


def get_depth(point_left, point_right, Q):
	d = point_left[0] - point_right[0]
	b = 1 / Q[3, 2]
	f = Q[2, 3]
	return f * b / d


def is_gray(image):
	"""
	判断图像是否为灰度图
	"""
	return len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)


def get_template(image, template):
	# 读取模板图像和待匹配图像
	if not is_gray(image):
		img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		img_gray = image
	if not is_gray(template):
		template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	else:
		template_gray = template
	img_threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	template_threshold = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	HEIGHT, WIDTH = img_gray.shape[:]
	height, width = template_gray.shape[:]
	# 模板匹配起始位置
	# 设置模板匹配的起始位置
	start_x = int(0.1 * WIDTH)
	start_y = int(0.1 * HEIGHT)
	# 限制模板匹配的搜索范围
	roi = img_threshold[start_y:start_y + HEIGHT, start_x:start_x + WIDTH]
	
	# 进行模板匹配
	result = cv2.matchTemplate(roi, template_threshold, cv2.TM_SQDIFF)
	# 获取匹配结果中最大的相似度值和对应的位置
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
	
	# 计算匹配区域在原始图像中的坐标
	top_left = (min_loc[0] + start_x, min_loc[1] + start_y)
	bottom_right = (min_loc[0] + width + start_x, min_loc[1] + height + start_y)
	aoi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
	return aoi, top_left, bottom_right


img_left = cv2.imread(r"D:\fy.xie\fenx\fenx - General\Ubei\Stereo\stereo_img\1000w_edge\Image_5.bmp")
img_right = cv2.imread(r"D:\fy.xie\fenx\fenx - General\Ubei\Stereo\stereo_img\1000w_edge\Image_6.bmp")
rect_left_image, rect_right_image, Q = StereoRectify.get_rectify(img_left, img_right, r'./config/calibration_parameters.json', r'./config/rectify_parameters.json')
for i in range(3):
	path = r"D:\fy.xie\fenx\fenx - General\Ubei\Stereo\stereo_img\1000w_edge\template" + str(i + 1) + ".png"
	template = cv2.imread(path)
	# 寻找匹配区域
	aoi_left, left_top_left, left_bottom_right = get_template(img_left, template)
	aoi_right, right_top_left, right_bottom_right = get_template(img_right, aoi_left)
	# 绘制匹配区域
	cv2.rectangle(img_left, left_top_left, left_bottom_right, (0, 0, 255), 1)
	cv2.rectangle(img_right, right_top_left, right_bottom_right, (0, 0, 255), 1)
	result = cv2.hconcat([img_left, img_right])
	# 匹配对应点
	depth = get_depth(left_bottom_right, right_bottom_right, Q)
	print(depth)
	cv2.imwrite(r"D:\fy.xie\fenx\fenx - General\Ubei\Stereo\stereo_img\1000w_edge\result" + str(i + 1) + ".png", result)
	i += 1
# # 显示匹配结果
# cv2.namedWindow('Result', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
