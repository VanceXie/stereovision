from math import ceil

from tools.SerializeTools import *
from tools.ImageOperate import *

img_path = r'D:\fy.xie\fenx\fenx - General\Ubei\Stereo\stereo_img\1000w_25'
# resize_image('./images/image_test', scale_x=0.25, scale_y=0.25)
left_image = cv2.imread(os.path.join(img_path, 'Image_35.bmp'))
right_image = cv2.imread(os.path.join(img_path, 'Image_36.bmp'))

imgcat_source = cat2images(left_image, right_image)
HEIGHT = left_image.shape[0]
WIDTH = left_image.shape[1]
imageSize = (WIDTH, HEIGHT)
cv2.imwrite(os.path.join(img_path, 'imgcat_out.png'), imgcat_source)

calibrationParamsDict = json2config('./config/calibration_parameters.json')

for var_name, var_val in calibrationParamsDict.items():
	# 使用 exec() 函数可以根据字符串动态地创建变量名
	exec(f"{var_name} = {var_val}")

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

rectify_parameters = {
	"R_l"         : R_l.tolist(),
	"P_l"         : P_l.tolist(),
	"R_r"         : R_r.tolist(),
	"P_r"         : P_r.tolist(),
	"Q"           : Q.tolist(),
	"validPixROI1": validPixROI1,
	"validPixROI2": validPixROI2
}

config2json(rectify_parameters, './config/rectify_parameters.json')

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

imgcat_out = cat2images(rect_left_image, rect_right_image)

cv2.imwrite(os.path.join(img_path, 'imgcat_out.png'), imgcat_out)
cv2.imwrite(os.path.join(img_path, 'rect_left_image.png'), rect_left_image)
cv2.imwrite(os.path.join(img_path, 'rect_right_image.png'), rect_right_image)

# 鼠标点击事件
count = 0
calibration_params = [Q[2, 3], Q[3, 2]]
x1, y1 = 0, 0
x2, y2 = 0, 0
X, Y, Z = 0, 0, 0


def on_mouse(event, x, y, flags, param):
	global x1, x2, y1, y2, count, X, Y, Z
	if event == cv2.EVENT_LBUTTONDOWN:
		count += 1
		if count % 2 == 1:
			x1, y1 = x, y
			print(f'点{ceil(count / 2)}在左图：{x, y} ')
		elif count % 2 == 0:
			x2, y2 = x, y
			print(f'点{count / 2}在右图：{x - 3840, y} ')
			d = x2 - x1 - 3840
			f = param[0]
			b = -1 / param[1]
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


'''显示图片'''
# def on_press(event):
#     global x1, x2, y1, y2, count, X, Y, Z
#     count += 1
#     if count % 2 == 1:
#         x1, y1 = event.x, event.y
#         print(f'点{ceil(count / 2)}在左图：{event.x, event.y} ')
#     elif count % 2 == 0:
#         x2, y2 = event.x, event.y
#         print(f'点{count / 2}在右图：{event.x - 5472, event.y} ')
#         d = x2 - x1 - 5472
#         f = calibration_params[0]
#         b = -1 / calibration_params[1]
#         if count % 4 == 0:
#             Z_TEMP = f * b / d
#             X_TEMP = x1 * b / d
#             Y_TEMP = y1 * b / d
#             print(f'点{count / 2}在空间中{X_TEMP, Y_TEMP, Z_TEMP}')
#             # S = sqrt((X - X_TEMP) ** 2 + (Y - Y_TEMP) ** 2 + (Z - Z_TEMP) ** 2)
#             # print(f'两点在空间中的长度为{S} mm')
#
#             dZ = Z - Z_TEMP
#             print(f'两点深度差为{dZ}mm')
#         else:
#             Z = f * b / d
#             X = x1 * b / d
#             Y = y1 * b / d
#             print(f'点{count / 2}在空间中{X, Y, Z}')

ima = cv2.imread(os.path.join(img_path, 'imgcat_out.png'))
cv2.namedWindow("disparity", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.imshow("disparity", ima)
cv2.setMouseCallback("disparity", on_mouse, calibration_params)

if cv2.waitKey(0) == ord("q"):
	cv2.destroyAllWindows()
