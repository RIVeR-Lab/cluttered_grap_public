#!/usr/bin/env python3
import rospy
import numpy as np
from matplotlib import pyplot as plt

import cv2 as cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cluttered_grasp.srv import segment_plane, segment_planeResponse

from sklearn.cluster import KMeans, MiniBatchKMeans

class SegmentOutPlane:

    def __init__(self):
        rospy.init_node("segment_plane")
        self.NUM_GROUPS = 7
        self.br = CvBridge()
        self.current_img = None

        image_sub = rospy.Subscriber(
            '/averaged_depth_images', Image, self._image_callback)
        self.superpixel_srv = rospy.Service('/segment_plane', segment_plane, self._call)

    def segment_out_plane(self):
        img = self.current_img
        stacked_img = np.stack((img,)*3, axis=-1)
        depth_flat = stacked_img.reshape(stacked_img.shape[0] * stacked_img.shape[1], stacked_img.shape[2])

        clusters = MiniBatchKMeans(self.NUM_GROUPS)

        kmeans = clusters.partial_fit(depth_flat)
        data_kmeans = kmeans.predict(depth_flat)
        data_kmeans = data_kmeans.reshape(stacked_img.shape[:2])

        num_pix_per_group = []
        for group in range(self.NUM_GROUPS):
            num_pix_per_group.append(np.sum((data_kmeans == group)))

        anom_group_num = num_pix_per_group.index(max(num_pix_per_group))
        plane_mask = data_kmeans != anom_group_num
        return plane_mask #.astype(np.int32)

    # update the current image
    def _image_callback(self, msg):
        self.current_img = self.br.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def _call(self, req):
        plane_mask = self.segment_out_plane()
        plane_mask = plane_mask.astype(np.float32)
        plane_mask = self.br.cv2_to_imgmsg(
            plane_mask, "passthrough")  # turn into bytes 32FC1

        return segment_planeResponse(plane_mask)

if __name__ == "__main__":
    sop = SegmentOutPlane()
    rospy.spin()
