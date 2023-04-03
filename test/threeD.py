import numpy as np


# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]
    
    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)
    
    points_ = np.hstack((points_1, points_2, points_3))
    
    return points_


def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols
    
    points_ = hw3ToN3(points_3d).astype(np.int16)
    colors_ = hw3ToN3(colors).astype(np.int64)
    
    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)
    # rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)
    rgb = blue + green + red
    
    # 将坐标+颜色叠加为点云数组
    pointcloud = np.hstack((points_, red / 255., green / 255., blue / 255.)).astype(np.float64)
    
    # 删掉一些不合适的点
    # X = pointcloud[:, 0]
    # Y = pointcloud[:, 1]
    # Z = pointcloud[:, 2]
    #
    # remove_idx1 = np.where(Z <= 0)
    # remove_idx2 = np.where(Z > 1000)
    # remove_idx3 = np.where(X > 1000)
    # remove_idx4 = np.where(X < -1000)
    # remove_idx5 = np.where(Y > 1000)
    # remove_idx6 = np.where(Y < -1000)
    # remove_idx = np.hstack(
    #         (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))
    #
    # pointcloud_1 = np.delete(pointcloud, remove_idx, 0)
    
    return pointcloud
