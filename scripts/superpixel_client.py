#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cluttered_grasp.srv import superpixel, superpixelResponse
import cv2 as cv
from matplotlib import pyplot as plt

# Authored by Gary Lvov

class SuperPixelClient:
    def __init__(self):
        rospy.init_node("superpixel_client")
        self.br = CvBridge()
        self.get_superpixels()

    def get_superpixels(self):
        rospy.wait_for_service("/superpixel")
        try:
            get_image = rospy.ServiceProxy("/superpixel", superpixel)
            img = get_image().image # get the image in bytes format
            img = self.br.imgmsg_to_cv2(img)
            plt.figure()
            plt.imshow(img)
            plt.show()

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

if __name__=="__main__":
    spc = SuperPixelClient()
    rospy.spin()