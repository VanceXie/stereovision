# author: young
import cv2

img_left_rectified = cv2.imread('./images/rect_left_image.png')
img_right_rectified = cv2.imread('./images/rect_right_image.png')

imgL = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2GRAY)

numDisparities = 111
blockSize = 10
disparity = None


def config_num(x):
    global numDisparities, blockSize, disparity
    blockSize = cv2.getTrackbarPos('blockSize', 'BM')
    stereo = cv2.StereoSGBM_create(numDisparities=(x + 1) * 16, blockSize=2 * blockSize + 5)
    disparity = stereo.compute(imgL, imgR)
    
    # 计算出的视差是CV_16S格式-16位有符号整数（-32768…32767）
    disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    disparity = cv2.applyColorMap(disparity, 2)


def config_block(y):
    global numDisparities, blockSize, disparity
    numDisparities = cv2.getTrackbarPos('num', 'BM')
    stereo = cv2.StereoSGBM_create(numDisparities=(numDisparities + 1) * 16, blockSize=2 * y + 5)
    disparity = stereo.compute(imgL, imgR)
    
    # 计算出的视差是CV_16S格式-16位有符号整数（-32768…32767）
    disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    disparity = cv2.applyColorMap(disparity, 2)


cv2.namedWindow('BM', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('num', 'BM', 111, 255, config_num)
cv2.createTrackbar('blockSize', 'BM', 10, 20, config_block)

'''
numDisparities：disparity的搜索范围，即最大差异减去最小差异。该值始终大于零。在当前的实现中，这个参数必须可以被16整除。
blockSize：块的线性大小。大小应该是奇数（因为块位于当前像素的中心）。更大的块大小意味着更平滑，但不太准确的视差图。较小的块大小会给出更详细的视差图，但算法找到错误对应的几率更高。一般在3到11之间。
'''

while 1:
    cv2.imshow('BM', disparity)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
