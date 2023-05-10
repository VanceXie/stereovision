from math import ceil

from tools.SerializeTools import *
from tools.ImageOperate import *


def get_rectify(left_image, right_image, calibration_json, rectify_json):
	HEIGHT, WIDTH = left_image.shape[:2]
	imageSize = (WIDTH, HEIGHT)
	
	calibration_config: dict = json2config(calibration_json)
	for var_name, var_val in calibration_config.items():
		# 使用 exec() 函数可以根据字符串动态地创建变量名,存在作用域问题
		# exec(f"{var_name} = {var_val}")
		# 将变量名和值存储到当前作用域的字典中
		globals()[var_name] = var_val
	
	R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(np.asarray(cameraMatrix_left),
																		  np.asarray(distCoeffs_left),
																		  np.asarray(cameraMatrix_right),
																		  np.asarray(distCoeffs_right),
																		  imageSize,
																		  np.asarray(R),
																		  np.asarray(T),
																		  alpha=1,
																		  flags=cv2.CALIB_ZERO_DISPARITY,
																		  newImageSize=imageSize)  # 计算旋转矩阵和投影矩阵
	
	# # 将返回参数（相机矫正参数）加入字典，并写入json配置文件
	rectify_config: dict = {
		"R_l"         : R_l.tolist(),
		"P_l"         : P_l.tolist(),
		"R_r"         : R_r.tolist(),
		"P_r"         : P_r.tolist(),
		"Q"           : Q.tolist(),
		"validPixROI1": validPixROI1,
		"validPixROI2": validPixROI2
	}
	
	config2json(rectify_config, rectify_json)
	
	# 左右图需要分别计算校正查找映射表以及重映射,计算校正查找映射表
	left_map_x, left_map_y = cv2.initUndistortRectifyMap(np.asarray(cameraMatrix_left),
														 np.asarray(distCoeffs_left),
														 R_l, P_l, imageSize, cv2.CV_32FC1)
	right_map_x, right_map_y = cv2.initUndistortRectifyMap(np.asarray(cameraMatrix_right),
														   np.asarray(distCoeffs_right),
														   R_r, P_r, imageSize, cv2.CV_32FC1)
	# 重映射
	rect_left_image = cv2.remap(left_image, left_map_x, left_map_y, cv2.INTER_CUBIC)
	rect_right_image = cv2.remap(right_image, right_map_x, right_map_y, cv2.INTER_CUBIC)
	return rect_left_image, rect_right_image, Q


def on_mouse(event, x, y, flags, param):
	global x1, x2, y1, y2, count, X, Y, Z
	if event == cv2.EVENT_LBUTTONDOWN:
		count += 1
		if count % 2 == 1:
			x1, y1 = x, y
			print(f'点{ceil(count / 2)}在左图：{x, y} ')
		elif count % 2 == 0:
			x2, y2 = x, y
			print(f'点{count / 2}在右图：{x - param[1], y} ')
			d = abs(x2 - x1 - param[1])
			f = param[0][2, 3]
			b = 1 / param[0][3, 2]
			if count % 4 == 0:
				Z_TEMP = f * b / d
				X_TEMP = x1 * b / d
				Y_TEMP = y1 * b / d
				print(f'点{count / 2}在空间中{X_TEMP, Y_TEMP, Z_TEMP}')
				# S = sqrt((X - X_TEMP) ** 2 + (Y - Y_TEMP) ** 2 + (Z - Z_TEMP) ** 2)
				# print(f'两点在空间中的长度为{S} mm')
				
				dZ = Z - Z_TEMP
				print(f'两点深度差为{dZ}mm')
			else:
				Z = f * b / d
				X = x1 * b / d
				Y = y1 * b / d
				print(f'点{count / 2}在空间中{X, Y, Z}')


def manual_select(left_image, right_image, calibration_json, rectify_json):
	"""
	:param left_image: path of left image
	:param right_image: path of right image
	:param calibration_json: path of calibration parameters to read
	:param rectify_json: path of rectify parameters to save
	:return:
	"""
	HEIGHT, WIDTH = left_image.shape[:2]
	rect_left_image, rect_right_image, Q = get_rectify(left_image, right_image, calibration_json, rectify_json)
	cv2.imwrite(r"C:\Users\vance\Desktop\left.png",rect_left_image)
	cv2.imwrite(r"C:\Users\vance\Desktop\right.png",rect_right_image)
	imgcat_out = cat2images(rect_left_image, rect_right_image)
	# 鼠标点击事件
	global x1, x2, y1, y2, count, X, Y, Z
	count = 0
	x1, y1 = 0, 0
	x2, y2 = 0, 0
	X, Y, Z = 0, 0, 0
	cv2.namedWindow("disparity", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
	cv2.imshow("disparity", imgcat_out)
	cv2.setMouseCallback("disparity", on_mouse, (Q, WIDTH))
	
	if cv2.waitKey(0) == ord("q"):
		cv2.destroyAllWindows()

#
# left_image = cv2.imread(r"D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_edge\Image_5.bmp")
# right_image = cv2.imread(r"D:\Fenkx\Fenkx - General\Ubei\Stereo\stereo_img\1000w_edge\Image_6.bmp")
# manual_select(left_image, right_image, r'./config/calibration_parameters.json', r'./config/rectify_parameters.json')
