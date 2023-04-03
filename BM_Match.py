# -*- coding: UTF-8 -*-
import json

import cv2
import numpy as np
import open3d as o3d

img_left_rectified = cv2.imread('./images/rect_left_image.png')
img_right_rectified = cv2.imread('./images/rect_right_image.png')

imgL = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2GRAY)

try:
    with open('./config/rectify_parameters.json', mode="r", encoding="utf-8") as file:
        rectify_parameters = json.load(file)
        Q = np.asarray(rectify_parameters['Q'])  # 即上文标定得到的 Q
        print("***Rectification parameters are loaded successfully!***\n")
except:
    print("Error: 没有找到文件或读取文件失败")

try:
    with open('./config/calibration_parameters.json', mode="r", encoding="utf-8") as file:
        calibration_parameters = json.load(file)
        cameraMatrix_left = np.asarray(calibration_parameters['cameraMatrix_left'])  # 即上文标定得到的 cameraMatrix_left
        cameraMatrix_right = np.asarray(calibration_parameters['cameraMatrix_right'])  # 即上文标定得到的 cameraMatrix_left
        print("***Rectification parameters are loaded successfully!***\n")
except:
    print("Error: 没有找到文件或读取文件失败")

numDisparities = 112
blockSize = 1

stereo = cv2.StereoBM_create(numDisparities=numDisparities * 16, blockSize=2 * blockSize + 5)
disparity = stereo.compute(imgL, imgR)
# disparity_show = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
# disparity_color = cv2.applyColorMap(disparity_show, 2)
# 视差是16位有符号格式，如StereoBM或StereoSGBM以及其他算法所计算的，则在此处使用之前应将其除以16（并缩放为浮点）
dis_real = disparity.astype(np.float32) / 16.

xyz = cv2.reprojectImageTo3D(dis_real, Q, handleMissingValues=True)
Z = np.where(xyz[:, :, 2] >= 200, 201, xyz[:, :, 2])
# Z=xyz[:, :, 2]
deepth_show = cv2.normalize(Z.astype(np.uint8), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8UC1)

deepth_color = cv2.applyColorMap(deepth_show, 2)


# 鼠标点击选点
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('点 (%d, %d) 的三维坐标 (%f, %f, %f)' % (x, y, xyz[y, x, 0], xyz[y, x, 1], xyz[y, x, 2]))


cv2.imshow("deepth_color", deepth_color)
cv2.setMouseCallback("deepth_color", onMouse, 0)

# 使用open3d库绘制点云
colorImage = o3d.geometry.Image(img_left_rectified)
depthImage = o3d.geometry.Image(Z.astype(np.float32))
rgbdImage = o3d.geometry.RGBDImage().create_from_color_and_depth(colorImage, depthImage, depth_scale=1, depth_trunc=200)

height, width = img_left_rectified.shape[0:2]
# 相机内参
intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, cameraMatrix_left)
# # 相机外参
# extrinsics = np.array([[1., 0., 0., 0.],
#                        [0., 1., 0., 0.],
#                        [0., 0., 1., 0.],
#                        [0., 0., 0., 1.]])
point_cloud = o3d.geometry.PointCloud().create_from_rgbd_image(rgbdImage, intrinsic=intrinsics)
o3d.io.write_point_cloud('PointCloud.pcd', point_cloud)
vis = o3d.visualization.VisualizerWithVertexSelection()
vis.create_window(window_name='Open3D', visible=True)
vis.add_geometry(point_cloud)
vis.run()
points = vis.get_picked_points()
vis.destroy_window()

# list(map(lambda item: print(item.index, item.coord), point))  # 内存开销大
for item in points:
    print(item.index, item.coord)

if cv2.waitKey(1) == ord("q"):
    cv2.destroyAllWindows()
# sys.exit(0)
