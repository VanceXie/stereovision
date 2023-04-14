import cv2
import numpy as np
import open3d as o3d

from threeD import DepthColor2Cloud

# 畸变矫正脚本

left_remap = cv2.imread('../images/rect_left_image.png')
right_remap = cv2.imread('../images/rect_right_image.png')
imgL_gray = cv2.cvtColor(left_remap, cv2.COLOR_BGR2GRAY)
imgR_gray = cv2.cvtColor(right_remap, cv2.COLOR_BGR2GRAY)
### 设置参数
# 块大小必须为奇数(3-11)
blockSize = 7
img_channels = 2
numDisparities = 1

param = {
    'preFilterCap'     : 15,  # 映射滤波器大小，默认15
    "minDisparity"     : 0,  # 最小视差
    "numDisparities"   : numDisparities * 16,  # 视差的搜索范围，16的整数倍
    "blockSize"        : blockSize,
    "uniquenessRatio"  : 10,  # 唯一检测性参数，匹配区分度不够，则误匹配(5-15)
    "speckleWindowSize": 0,  # 视差连通区域像素点个数的大小（噪声点）(50-200)或用0禁用斑点过滤
    "speckleRange"     : 1,  # 认为不连通(1-2)
    "disp12MaxDiff"    : 2,  # 左右一致性检测中最大容许误差值
    "P1"               : 8 * blockSize ** 2,  # 值越大，视差越平滑，相邻像素视差+/-1的惩罚系数
    "P2"               : 32 * blockSize ** 2,  # 同上，相邻像素视差变化值>1的惩罚系数
    # 'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
}
## 开始计算深度图
left_matcher = cv2.StereoSGBM_create(**param)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

left_disp = left_matcher.compute(imgL_gray, imgR_gray)
right_disp = right_matcher.compute(imgR_gray, imgL_gray)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
# sigmaColor典型范围值为0.8-2.0
wls_filter.setLambda(8000.)
wls_filter.setSigmaColor(1.3)
wls_filter.setLRCthresh(24)
wls_filter.setDepthDiscontinuityRadius(3)

filtered_disp = wls_filter.filter(left_disp, imgL_gray, disparity_map_right=right_disp)

disp = cv2.normalize(filtered_disp, filtered_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

Q = np.asarray(
        [[1.0, 0.0, 0.0, -1782.2614288330078], [0.0, 1.0, 0.0, -1240.004898071289], [0.0, 0.0, 0.0, 10097.130041623303],
         [0.0, 0.0, -0.034674583504178125, 0.0]])
threeD = cv2.reprojectImageTo3D(disp, Q)
pointcloud = DepthColor2Cloud(threeD, left_remap)

# 转换为open3d的点云数据
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:])
o3d.visualization.draw_geometries_with_editing([pcd], window_name="3D", width=3840, height=2748)
