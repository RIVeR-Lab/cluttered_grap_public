#!/usr/bin/env python3
import os
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
# Authored by Gary Lvov and Nathaniel Hanson
class Associate:
    def __init__(self):
        rospy.init_node("hsi_rgb_reg")

        self.br = CvBridge()
        self.current_img = None

        self.raw_cube = None
        self.MIN_MATCH_COUNT = 10
        self.NUM_WAVELENGTHS = 273
        self.GRIPPER_CROP_THRESH = 600

        image_sub = rospy.Subscriber('/rgb/image_raw', Image, self._image_callback) 
        self.associate_srv = rospy.Service('/associate', several, self._call)

        self.cumulative_hsi = None
        self.raw_cubes = []
        self.perspective_transforms = []

        self.pkg_path = rospkg.RosPack().get_path("cluttered_grasp") # package path
        dark_img = open_image(self.pkg_path + '/scripts/darkReference.hdr')
        white_img = open_image(self.pkg_path + '/scripts/whiteReference.hdr')
        self.data_dark = np.array(dark_img.load()).reshape(640,273)
        self.data_white = np.array(white_img.load()).reshape(640,273)

        self.sift = cv.SIFT_create()
        self.flann = cv.FlannBasedMatcher(dict(algorithm = 1, trees = 5), dict(checks = 50))

    def pad_or_clip(self, array, xx, yy):
        """
        :param array: numpy array
        :param xx: desired height
        :param yy: desirex width
        :return: padded array
        """
        h = array.shape[0]
        w = array.shape[1]
        # Check if array is too large
        if h > xx or w > yy:
            return array[:xx,:yy]
        
        a = (xx - h) // 2
        aa = xx - a - h

        b = (yy - w) // 2
        bb = yy - b - w

        return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


    def denoise_data(self, data):
        signal = calc_stats(data)
        noise = noise_from_diffs(data[0:data.shape[0]//16, 0:data.shape[1]//16, :])
        mnfr = mnf(signal, noise)
        return mnfr.denoise(data, snr=10).astype(np.float32, copy=False)

    # process the hyperspectral image to possess the correct perspective transform relative to rgb image
    def associate_hsi_rgb (self, hsi_paths):
        # Process the Center Image first
        if(len(hsi_paths) == 3):
            hsi_paths[0], hsi_paths[1] = hsi_paths[1], hsi_paths[0]

        # Do all proceses on the same image - converted to greyscale
        kinect_img = cv.cvtColor(self.current_img, cv.COLOR_BGR2GRAY)

        #kinect_img = kinect_img[:self.GRIPPER_CROP_THRESH, :] # crop out gripper

        h, w = kinect_img.shape
        hsi = [] # Keep track of all greyscale indicative images

        path = self.pkg_path + "/images/vest/"
        cv.imwrite(os.path.join(path , "kinect_image_initial.png"), kinect_img)

        # Load each cube from the list of filepaths to create an indicative greyscale image
        # Keep track of cubes and images
        for cube_index in range(len(hsi_paths)):
            print(hsi_paths[cube_index])
            if(os.path.exists(hsi_paths[cube_index])):
                self.raw_cubes.append(np.load(hsi_paths[cube_index]))
                self.raw_cubes[cube_index] = self.normalize(self.raw_cubes[cube_index])
                img = self.create_image_from_cube(self.raw_cubes[cube_index])
                hsi.append(img)
                
        # Pre-Process the images to have better feature matches for homography
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsi_improved = []
        for x in range(len(hsi_paths)):
            dst = cv.fastNlMeansDenoising(hsi[x],None,10,9,21)
            cl1 = clahe.apply(dst)
            hsi_improved.append(cl1.copy())

        cv.imwrite(os.path.join(path , "hsi1.png"), hsi_improved[0])
        cv.imwrite(os.path.join(path , "hsi2.png"), hsi_improved[1])
        cv.imwrite(os.path.join(path , "hsi3.png"), hsi_improved[2])

        assert len(hsi) >= 1, "Need at least one image"

        # Start the stitched image with the first image in the sequence
        if(self.cumulative_hsi is None):
            self.cumulative_hsi = hsi[0]
            M = self.determine_homography(hsi_improved[0], hsi_improved[0])
            comb, H = self.warpTwoImages(hsi_improved[0], hsi_improved[0], M)
       
        # Progressively stitch the HSI together, keeping track of perspective transforms
        for idx, image in enumerate(hsi_improved[1:]):
            M = self.determine_homography(image, self.cumulative_hsi)
            self.perspective_transforms.append(M)
            self.cumulative_hsi, H = self.warpTwoImages(self.cumulative_hsi, image, M)
            cv.imwrite(os.path.join(path , ("stitch" + str(idx) + ".png")), image)

        # Determine the transform between stitched HSI and kinect image    
        M = self.determine_homography(self.cumulative_hsi, kinect_img)
        self.perspective_transforms.append(M)
        self.cumulative_hsi = cv.warpPerspective(self.cumulative_hsi, M, (w, h))

        cv.imwrite(os.path.join(path , "cumulative_hsi_final.png"), self.cumulative_hsi)

        # Normalize all loaded cubes, then rotate to be in same orientation as kinect img
        for cube in range(len(self.raw_cubes)):
            adapted = np.flip(self.raw_cubes[cube],axis=0)
            adapted = np.rot90(adapted)
            print(f'Processing raw cube # {cube}')
            np.nan_to_num(adapted, copy=False, nan=0.0)
            self.raw_cubes[cube] = adapted
        
        '''
        Transform the hyperspectral cubes according to previously obtained homography,
        creating a 1-1 association between pixels in the RGB image and pixels in the 
        hyperspectral cube.
        '''
        initial_warp = []
        acc_cube = []
        for cube in range(len(self.raw_cubes) - 1):
            
            initial_warp.append(self.warpTwoImages(self.raw_cubes[cube][:,:,0], 
                                    self.raw_cubes[cube + 1][:,:,0], 
                                    self.perspective_transforms[cube])[0])
            
            acc_cube.append(np.zeros((*(initial_warp[cube].shape), self.NUM_WAVELENGTHS), dtype=np.float32))
                
            for wave in range(self.NUM_WAVELENGTHS):
                if(len(acc_cube) == 1):
                    comp = self.raw_cubes[0][:,:,wave]
                else:
                    comp = acc_cube[cube - 1][:,:,wave]
                    
                
                tData, _ = self.warpTwoImages(comp, 
                                                        self.raw_cubes[cube + 1][:,:,wave],
                                                        self.perspective_transforms[cube])
                
                acc_cube[cube][:,:,wave] = self.pad_or_clip(tData, acc_cube[cube].shape[0], acc_cube[cube].shape[1])
                
        acc_cube.append(np.zeros((*kinect_img.shape, self.NUM_WAVELENGTHS), dtype=np.float32))

        for wave in range(self.NUM_WAVELENGTHS):
            acc_cube[-1][:,:,wave] = cv.warpPerspective(acc_cube[-2][:,:,wave],
                                                        self.perspective_transforms[-1],
                                                        acc_cube[-1].shape[:2][::-1])

        raw_cubes = []
        acc_cube[:-1] = []
        acc_cube[0].resize((*(self.current_img.shape)[:2], self.NUM_WAVELENGTHS))
        # Remove errant NaN values
        np.nan_to_num(acc_cube[0], copy=False, nan=0.0)
        np.save(self.pkg_path + "/associated_normalized_cube.npy", acc_cube[0])
        cv.imwrite(os.path.join(path , "final_cube.png"), acc_cube[0][:,:,40])
        print("saved cube at: " + self.pkg_path + "/associated_normalized_cube.npy")



    # Given two images, returns the perspective transform between the child and parent
    # https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
    def determine_homography(self, child: np.ndarray, parent: np.ndarray) -> np.ndarray:
        kp1, des1 = self.sift.detectAndCompute(child, None)
        kp2, des2 = self.sift.detectAndCompute(parent, None)
        matches = self.flann.knnMatch(des1,des2,k=2)
        good = []
        for m, n in matches:
            if m.distance < .7 * n.distance: #tutorial has 0.7*n.distance, increase thresh for more matches
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
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel())
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel())
        t = [-xmin,-ymin]
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

        H =  Ht.dot(M)
        result = cv.warpPerspective(parent, H, (xmax-xmin, ymax-ymin))
        result[t[1]:h1+t[1],t[0]:w1+t[0]] = child
        return result, H

    def normalize(self, raw_cube: np.ndarray) -> np.ndarray:
        # Normalize the cube based off of radiometric calibration
        cal_cube = np.zeros_like(raw_cube,dtype=np.float32)
        for i in range(raw_cube.shape[0]):
            cal_cube[i,:] = np.clip((raw_cube[i,:] - self.data_dark) / (self.data_white - self.data_dark),0,1)
        return cal_cube  

    # Given a cube, create an indicative greyscale image
    def create_image_from_cube(self, raw_cube: np.ndarray) -> np.ndarray:
        # Normalize the cube based off of radiometric calibration
        cal_cube = raw_cube * 255
            
        # copy indicative bands from hsi cube into rgb values for corresponding pixels
        adapted = np.dstack((cal_cube[:,:,140], cal_cube[:,:,63], cal_cube[:,:,36]))
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
        print(req.data)
        self.associate_hsi_rgb(req.data)
        return severalResponse(True)

if __name__=="__main__":
    a = Associate()
    rospy.spin()