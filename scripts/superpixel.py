#!/usr/bin/env python3
import cv2
import rospy
import numpy as np
from matplotlib import pyplot as plt

import cv2 as cv
from skimage.segmentation import slic, chan_vese, mark_boundaries
from spectral import open_image
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cluttered_grasp.srv import superpixel, superpixelResponse

# Authored by Gary Lvov

class SuperPixel:

    def __init__(self):
        rospy.init_node("superpixel")

        self.br = CvBridge()
        self.current_img = None

        image_sub = rospy.Subscriber(
            '/rgb/image_raw', Image, self._image_callback)
        # image_sub = rospy.Subscriber('/image_publisher/image_raw', Image,
        self.superpixel_srv = rospy.Service('/superpixel', superpixel, self._call)

    def superpixel(self):
        img = self.current_img
        img = img[:,:,:3]
        segments = slic(img, n_segments=1000, sigma=5, compactness=20)

        fig = plt.figure("Superpixels -- %d segments" % (1000))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(img, segments))
        plt.axis("off")
        # show the plots
        plt.show()
        plt.imsave("superpixels.png", mark_boundaries(img, segments))
        return segments #.astype(np.int32)

    # update the current image
    def _image_callback(self, msg):
        self.current_img = self.br.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def _call(self, req):
        superpixeled_img = self.superpixel()
        superpixeled_img = superpixeled_img.astype(np.float32)
        np.save('/home/river/superpixels.npy', superpixeled_img)
        superpixeled_img = self.br.cv2_to_imgmsg(
            superpixeled_img, "32FC1")  # turn into bytes
        return superpixelResponse(superpixeled_img)

if __name__ == "__main__":
    sp = SuperPixel()
    rospy.spin()
