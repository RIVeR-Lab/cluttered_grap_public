#!/usr/bin/env python3
import numpy as np
import rospy
import rospkg
import cv2 as cv
from spectral import *
from matplotlib import pyplot as plt
from cv_bridge import CvBridge

from sklearn.cluster import KMeans, MiniBatchKMeans
from cluttered_grasp.srv import pure_kmeans, pure_kmeansResponse
from copy import deepcopy

# Authored by Gary Lvov and Nathaniel Hanson

class Pure:
    def __init__(self):
        rospy.init_node("find_pure_regions_kmeans")
        self.NUM_WAVELENGTHS = 273
        self.NUM_GROUPS = 3
        self.PURE_THRES = .10

        self.br = CvBridge()

        self.pkg_path = rospkg.RosPack().get_path("cluttered_grasp")

        self.find_pure = rospy.Service("/pure_with_kmeans", pure_kmeans, self._call)

    def find_pure_regions(self, superpixels):
        data_cube = np.load(self.pkg_path + "/associated_normalized_cube.npy")
        np.nan_to_num(data_cube, copy=False, nan=0.0)
        noise = noise_from_diffs(data_cube[500:700, 1000: 1200, :])
        signal = calc_stats(data_cube)
        mnfr = mnf(signal, noise)
        reduced = mnfr.reduce(data_cube, num=10)

        data_2d= reduced.reshape(reduced.shape[0]*reduced.shape[1], reduced.shape[2])
        # Only use the region of interest here
        trainable = reduced[300:1000,750:1750,:]
        trainable = trainable.reshape(trainable.shape[0]*trainable.shape[1], trainable.shape[2])
        clusters = MiniBatchKMeans(self.NUM_GROUPS, verbose=1)

        kmeans = clusters.partial_fit(trainable)
        data_kmeans = kmeans.predict(data_2d)
        data_kmeans = data_kmeans.reshape(reduced.shape[:2])

        num_pix_per_group = []
        for group in range(self.NUM_GROUPS):
            plt.figure()
            plt.imshow(data_kmeans == group)
            num_pix_per_group.append(np.sum((data_kmeans == group)[300:1000,750:1750]))
        
        anom_group_num = num_pix_per_group.index(min(num_pix_per_group))

        grouped_kmeans = []

        for superpixel in np.unique(superpixels):
            clusters = []
            for group in range(self.NUM_GROUPS):
                clusters.append(np.sum((data_kmeans == group)[superpixels == superpixel]))
            total_pix = sum(clusters)
            if clusters[anom_group_num]/total_pix > self.PURE_THRES:
                mode = anom_group_num
            else:
                mode = clusters.index(max(clusters))
                
            grouped_kmeans.append(mode)
        
        pure_reg = np.zeros_like(superpixels)

        for pix, mode in enumerate(grouped_kmeans):
            if (grouped_kmeans[pix] == anom_group_num):
                pure_reg[superpixels == pix] = 0
            else:
                pure_reg[superpixels == pix] = 1
        plt.figure()
        plt.imshow(pure_reg)
        plt.title('')
        plt.show()
        return pure_reg

    def perform_rx(self, data, n_pixels):
        rospy.wait_for_service("/rx")
        try:
            get_rx = rospy.ServiceProxy("/rx", rx)
            rx_result = get_rx(data.flatten(), n_pixels, self.NUM_WAVELENGTHS)
            return rx_result.data
            
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
    
    def _call(self, req):
        superpixels = req.superpixels
        superpixels = self.br.imgmsg_to_cv2(superpixels)
        pure_reg = self.find_pure_regions(superpixels)
        pure_reg = self.br.cv2_to_imgmsg(pure_reg, "32FC1")

        return pure_kmeansResponse(pure_reg)

if __name__=="__main__":
    p = Pure()
    rospy.spin()