#!/usr/bin/env python3
import os
import rospy
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Authored by Gary Lvov and Nathaniel Hanson

class Average:
    def __init__(self):

        rospy.init_node("average_images")

        self.br = CvBridge()
        self.current_img = None
        self.ROLLING_SIZE = 10
        self.pkg_path = rospkg.RosPack().get_path("cluttered_grasp") # package path
        self.rolling_imgs = []
        image_sub = rospy.Subscriber('/depth_to_rgb/image_raw', Image, self._image_callback) 
        self.image_pub = rospy.Publisher('averaged_depth_images', Image, queue_size=10)
        self.publish_average()

    def publish_average(self):
        while not (rospy.is_shutdown()):
            if self.current_img is not None and self.rolling_imgs != []:
                base = np.mean(self.rolling_imgs, axis=2).astype(np.uint16)
                self.image_pub.publish(self.br.cv2_to_imgmsg(base, encoding="16UC1"))

    def _image_callback(self, msg):
        self.current_img = self.br.imgmsg_to_cv2(msg)
        np.save('/home/river/depth.npy',self.current_img)
        if self.rolling_imgs == []:
            self.rolling_imgs = np.zeros((*self.current_img.shape, self.ROLLING_SIZE), dtype=np.uint16)
        else:
            self.rolling_imgs = np.roll(self.rolling_imgs, 1, axis=2)
        self.rolling_imgs[:,:,0] = self.current_img

if __name__=="__main__":
    a = Average()
    rospy.spin()