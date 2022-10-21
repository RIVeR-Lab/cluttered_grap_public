#!/usr/bin/env python3
import rospy
import rospkg
import numpy as np

import cv2 as cv
from cluttered_grasp.srv import segment_plane
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
# Authored by Gary Lvov
rospy.init_node("test_plane_seg")
br = CvBridge()

def get_plane():
    rospy.wait_for_service("/segment_plane")
    try:
        get_seg = rospy.ServiceProxy('/segment_plane', segment_plane)
        seg_img = get_seg().image
        seg_img = br.imgmsg_to_cv2(seg_img)
        return seg_img

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

filtered_img = cv.imread("img.png")
plane_seg = get_plane()
print(plane_seg)
print(plane_seg.dtype)
plt.figure()
plt.imshow(plane_seg)
plt.show()

plane_seg = cv.convertScaleAbs(plane_seg)
print(plane_seg)
print(plane_seg.dtype)
plt.figure()
plt.imshow(plane_seg)
plt.show()

indicies = cv.bitwise_and(filtered_img,filtered_img,mask = plane_seg)
print(indicies.shape)


plt.figure()
plt.imshow(indicies)
plt.show()
plt.imsave("mask_out_plane.png", indicies, cmap="viridis")