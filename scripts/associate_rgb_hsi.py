#!/usr/bin/env python3
import rospy
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from spectral import *

import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cluttered_grasp.srv import several, severalResponse

from typing import Tuple

from pprint import pprint


class Associate:
    def __init__(self):
        rospy.init_node("hsi_rgb_reg")

        self.br = CvBridge()
        self.current_img = None

        self.raw_cube = None
        self.MIN_MATCH_COUNT = 50

        image_sub = rospy.Subscriber('/rgb/image_raw', Image, self._image_callback) 
        # image_sub = rospy.Subscriber('/image_publisher/image_raw', Image, self._image_callback) for debug
        self.restart_srv = rospy.Service('/hsi_rgb_reg/associate_from_cube_paths', several, self._call)

        self.cumulative_hsi = None
        self.perspective_transforms = []

        pkg_path = rospkg.RosPack().get_path("cluttered_grasp") # package path
        dark_img = open_image(pkg_path + '/scripts/darkReference.hdr')
        white_img = open_image(pkg_path + '/scripts/whiteReference.hdr')
        self.data_dark = np.array(dark_img.load()).reshape(640,273)
        self.data_white = np.array(white_img.load()).reshape(640,273)

        self.MIN_MATCH_COUNT = 10
        self.sift = cv.SIFT_create(2000)
        self.flann = cv.FlannBasedMatcher(dict(algorithm = 1, trees = 5), dict(checks = 50))

    # process the hyperspectral image to possess the correct perspective transform relative to rgb image
    def associate_hsi_rgb (self, hsi_paths):
        kinect_img = self.current_img
        h, w, c = kinect_img.shape
        hsi = []

        # Process each cube from the list of filepaths to create an indicative greyscale image, then add it to the list of images
        for cube_path in hsi_paths:
            print(cube_path)
            img = self.create_image_from_cube(np.load(cube_path))
            hsi.append(img)


        assert len(hsi) >= 1, "Need at least one image"

        if(self.cumulative_hsi is None):
            self.cumulative_hsi = hsi[0]
            hsi.remove(hsi[0])
            M = self.determine_homography(hsi[0], hsi[0])
            comb, H = self.warpTwoImages(hsi[0], hsi[0], M)

        for image in hsi:
            M = self.determine_homography(image, self.cumulative_hsi)
            self.cumulative_hsi, H = self.warpTwoImages(self.cumulative_hsi, image, M)
            self.perspective_transforms.append(H)
            
        M = self.determine_homography(self.cumulative_hsi, kinect_img)
        self.perspective_transforms.append(M)
        self.cumulative_hsi = cv.warpPerspective(self.cumulative_hsi, M, (w, h))
        overlay = cv.cvtColor(kinect_img, cv.COLOR_BGR2GRAY)  + self.cumulative_hsi
        
        cv.namedWindow("Association Results", cv.WINDOW_NORMAL)
        cv.imshow("Association Results", overlay)
        cv.waitKey(0)

        cv.imwrite("ROSified-cumulative-results.png", self.cumulative_hsi)
        cv.imwrite("ROSified-results.png", overlay)
        pprint(self.perspective_transforms)

    # Given two images, returns the perspective transform between the child and parent
    # https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
    def determine_homography(self, child: np.ndarray, parent: np.ndarray) -> np.ndarray:
        kp1, des1 = self.sift.detectAndCompute(child, None)
        kp2, des2 = self.sift.detectAndCompute(parent, None)
        matches = self.flann.knnMatch(des1,des2,k=2)
        good = []
        for m, n in matches:
            if m.distance < .8 * n.distance: #tutorial has 0.7*n.distance, increase thresh for more matches
                good.append(m)
        
        assert len(good) > self.MIN_MATCH_COUNT, f"Needed to find {self.MIN_MATCH_COUNT} matches, instead only found {len(good)}"

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
        return M

    # Given two images and their perspective transform, combine the two images, and return the transform
    # https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
    def warpTwoImages(self, child: np.ndarray, parent: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''warp parent to child with homograph M'''
        h1,w1 = child.shape[:2]
        h2,w2 = parent.shape[:2]
        pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
        pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
        pts2_ = cv.perspectiveTransform(pts2, M)
        pts = np.concatenate((pts1, pts2_), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin,-ymin]
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

        H =  Ht.dot(M)
        result = cv.warpPerspective(parent, H, (xmax-xmin, ymax-ymin))
        result[t[1]:h1+t[1],t[0]:w1+t[0]] = child
        return result, H

    # Given a cube, create an indicative greyscale image
    def create_image_from_cube(self, raw_cube: np.ndarray) -> np.ndarray:
        # Normalize the cube based off of radiometric calibration
        cal_cube = np.zeros_like(raw_cube,dtype=np.float32)
        for i in range(raw_cube.shape[0]):
            cal_cube[i,:,:] = np.clip((raw_cube[i,:,:] - self.data_dark) / (self.data_white - self.data_dark),0,1)* 255
            
        # copy indicative bands from hsi cube into rgb values for corresponding pixels
        adapted = np.zeros([len(raw_cube), len(raw_cube[0]), 3], dtype=np.uint16)
        adapted[:,:,0] = cal_cube[:,:,140]
        adapted[:,:,1] = cal_cube[:,:,63] 
        adapted[:,:,2] = cal_cube[:,:,36]
        adapted = cv.convertScaleAbs(adapted) # Convert to correct datatype for visualization
        adapted = cv.cvtColor(adapted, cv.COLOR_BGR2GRAY) # convert to greyscale

        # Transform the image to match the RGB image
        adapted = cv.flip(adapted, 0)
        adapted = cv.rotate(adapted, cv.cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return adapted

    # update the current image
    def _image_callback(self, msg):
        self.current_img = self.br.imgmsg_to_cv2(msg)
    
    def _call(self, req):
        self.associate_hsi_rgb(req.data)
        return severalResponse(True)

if __name__=="__main__":
    a = Associate()
    rospy.spin()
