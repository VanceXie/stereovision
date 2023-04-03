# import numpy as np
# import matplotlib.pyplot as plt
#
# B = 28.84
# f = 10097.13
# d = 1650
# delta_d = np.arange(1,10,1)
# delta_D = B * f * (1 / d - 1 / (d + delta_d))
# plt.scatter(delta_d, delta_D)
# for i in range(len(delta_d)):
#     plt.annotate(delta_D[i], xy = (delta_d[i], delta_D[i]), xytext = (delta_d[i]-0.5, delta_D[i]+0.05)) # 这里xy是需要标记的坐标，xytext是对应的标签坐标
# plt.show()
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('../images/rect_left_image.png', cv.IMREAD_GRAYSCALE)  # referenceImage
img2 = cv.imread('../images/rect_right_image.png', cv.IMREAD_GRAYSCALE)  # sensedImage

# Initiate AKAZE detector
akaze = cv.AKAZE_create()
# Find the keypoints and descriptors with SIFT
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.27 * n.distance:
        good_matches.append([m])
# mat = []
# for i in range(20):
#     mat.append(good_matches[10 * i])


def get_coordinates_from_matches(matches: list, kp1: list, kp2: list):
    '''
    Created on Wed Aug 17 21:00:36 2022

    Input :
        matches : [type=DMatch, ...]
        kp1     : KeyPoints from query image, [type=KeyPoint, ...]
        kp2     : KeyPoints from train image, [type=KeyPoint, ...]

    Return:
        list1   : stores the matching keypoints for query image
        list2   : stores the matching keypoints for train image

    len(matches) == len(kp1) == len(kp2)
    @author: zqfeng
    '''
    # Initialize lists
    list_kp1 = []
    list_kp2 = []
    
    # For each match...
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat[0].queryIdx
        img2_idx = mat[0].trainIdx
        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))
    
    return list_kp1, list_kp2


list_kp1, list_kp2 = get_coordinates_from_matches(good_matches, kp1, kp2)
print(list_kp1,'\n',list_kp2)
# Draw matches
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('matches.png', img3)
