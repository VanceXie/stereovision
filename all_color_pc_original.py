# author: young
import json

import cv2
import numpy as np
import open3d as o3d

img = cv2.imread('./images/image_test/image_test_left.bmp', 1)
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_left_rectified = cv2.imread('./images/rect_left_image.png')  # 高度2748，宽度3840
img_right_rectified = cv2.imread('./images/rect_right_image.png')

imgL = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2GRAY)

num = 104
blockSize = 7
# numDisparities视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
# blockSize：SAD窗口大小，5~21之间为宜

try:
    with open('./config/rectify_parameters.json', mode="r", encoding="utf-8") as file:
        rectify_parameters = json.load(file)
        Q = np.asarray(rectify_parameters['Q'])  # 即上文标定得到的 Q
        print("***Rectification parameters are loaded successfully!***\n" * 3)
except:
    print("Error: 没有找到文件或读取文件失败")

S = cv2.StereoSGBM_create(numDisparities=16 * num, blockSize=blockSize)
dis = S.compute(imgL, imgR)

# 在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16
xyz = cv2.reprojectImageTo3D(dis, Q, handleMissingValues=True)
xyz = xyz * 16

dis_real = dis.astype(np.float32) / 16.
dis_show = cv2.normalize(dis, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
dis_show = cv2.medianBlur(dis_show, 9)
dis_color = cv2.applyColorMap(dis_show, 2)


# 鼠标点击事件
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('点 (%d, %d) 的三维坐标 (%f, %f, %f)' % (x, y, xyz[y, x, 0], xyz[y, x, 1], xyz[y, x, 2]))


cv2.imshow("disparity", dis_color)
cv2.setMouseCallback("disparity", onMouse, 0)

output_points = np.zeros((3840 * 2748, 6))

i = 0

b = -1 / Q[3, 2]
f = Q[2, 3]
cx = -Q[0, 3]
cy = -Q[1, 3]

for index, item in np.ndenumerate(dis_real):
    if item != 0 and item != -16:
        output_points[i][0] = b * (index[1] - cx) / item
        output_points[i][1] = b * (index[0] - cy) / item
        output_points[i][2] = b * f / item
        output_points[i][3] = img_color[index[0]][index[1]][0]
        output_points[i][4] = img_color[index[0]][index[1]][1]
        output_points[i][5] = img_color[index[0]][index[1]][2]
        i = i + 1


def creatp_output(vertices, filename):
    ply_header = '''ply
                    format ascii 1.0
                    element vertex %(vert_num)d
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                    '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


output_file = 'nb.ply'
creatp_output(output_points, output_file)
pcd = o3d.io.read_point_cloud(output_file)
o3d.visualization.draw_geometries([pcd])

cv2.waitKey(0)
cv2.destroyAllWindows()
