
import rospy

import cv2 as cv
from geometry_msgs.msg import Point
from cluttered_grasp.srv import superpixel
from cluttered_grasp.srv import nudge
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
# Authored by Gary Lvov
rospy.init_node("test_clutter_nudge")
br = CvBridge()

def get_superpixels():
        rospy.wait_for_service("/superpixel")
        try:
            get_image = rospy.ServiceProxy("/superpixel", superpixel)
            img = get_image().image # get the image in bytes format
            return img
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

superpixels = get_superpixels()

def nudge_test(targets, superpixels):
    rospy.wait_for_service("/nudge")
    try:
        get_nudge = rospy.ServiceProxy('/nudge', nudge)
        nudge_res_bool = get_nudge(targets, superpixels).res
        return nudge_res_bool
        
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

targets = []
p1 = Point()
p1.x = 900
p1.y = 900
p1.z = 10
targets.append(p1)
nudge_img = nudge_test(targets, superpixels)


