# -*- coding: UTF-8 -*-
import numpy as np


# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[10419.302339274147, 0.0, 2086.876873000707],
                                         [0.0, 10418.001100510719, 1241.3758892669393],
                                         [0.0, 0.0, 1.0]])
        # 右相机内参
        self.cam_matrix_right = np.array([[10419.302339274147, 0.0, 2086.7611187140883],
                                          [0.0, 10418.001100510719, 1241.038094771258],
                                          [0.0, 0.0, 1.0]])
        
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.023469418399945347, -1.9058520729207586, 8.758577828878978e-05,
                                       0.0006361688853099171, 0]])
        self.distortion_r = np.array([[-0.06652986416258569, 0.4166331158391988, -0.0002343185655796906,
                                       8.786231449294303e-05, 0]])
        
        # 旋转矩阵
        self.R = np.array([
            [
                0.9999985900366308,
                0.0013910681106536187,
                0.0009406669239747788
            ],
            [
                -0.0013930383856525256,
                0.999996830698076,
                0.002097149937300462
            ],
            [
                -0.0009377466643162244,
                -0.0020984573655290924,
                0.9999973585504507
            ]
        ])
        
        # 平移矩阵
        self.T = np.array([
            [
                -28.828846468322052
            ],
            [
                -0.278534700368728
            ],
            [
                -0.7800912243483239
            ]
        ])
        
        # 主点列坐标的差
        self.doffs = 0.0
        
        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False
    
    def setMiddleBurryParams(self):
        self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
                                         [0., 3997.684, 187.5],
                                         [0., 0., 1.]])
        self.cam_matrix_right = np.array([[3997.684, 0, 225.0],
                                          [0., 3997.684, 187.5],
                                          [0., 0., 1.]])
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        self.R = np.identity(3, dtype=np.float64)
        self.T = np.array([[-193.001], [0.0], [0.0]])
        self.doffs = 131.111
        self.isRectified = True
