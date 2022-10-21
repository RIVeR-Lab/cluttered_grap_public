#!/usr/bin/env python3
from lib2to3.pgen2.tokenize import TokenError
import numpy as np
import rospy
import rospkg
import cv2 as cv
from typing import List, Tuple
from matplotlib import pyplot as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from cluttered_grasp.srv import roi, roiResponse
from geometry_msgs.msg import Point
from copy import deepcopy
from scipy.spatial import ConvexHull

# Authored by Nathaniel Hanson

class RegionsOfInterest:
    def __init__(self):
        rospy.init_node("regions_of_interest")
        self.NUM_WAVELENGTHS = 273
        self.br = CvBridge()
        # Subscribe to average depth image
        depth_image_sub = rospy.Subscriber('averaged_depth_images', Image, self._image_callback)
        self.pkg_path = rospkg.RosPack().get_path("cluttered_grasp")
        self.find_roi = rospy.Service("/roi", roi, self._call)
        self.MAX_ANOMALIES = 20 #rospy.get_param('~max_anom', 10)


    
    def _call(self, req):
        data = self.br.imgmsg_to_cv2(req.errors)
        superpixels = self.br.imgmsg_to_cv2(req.superpixels)
        #### For every pixel, we want to consider the neighborhood
        # Strong pixels, with strong neighbors should become strong
        # Strong pixels, with weak neightbors should become weaker

        # Use superpixels to score local neighborhoods
        scores = np.zeros_like(superpixels)
        cell_scores = []
        reconstruct_mean = np.mean(data)
        for superpixel in np.unique(superpixels):
            mask = superpixels == superpixel
            if np.mean(data[mask]) < 0.00001:
                scores[mask] = 0
                cell_scores.append(0)
                continue
            use_data = data[mask]
            mean_score = (np.max(use_data)-np.median(use_data)) * np.std(use_data) * (np.sum(use_data > np.median(use_data)))**2
            scores[mask] = mean_score
            cell_scores.append(mean_score)
        top_scores = sorted(cell_scores, reverse=True)
        points = self.scores_to_points(top_scores, scores, self.MAX_ANOMALIES)
        points_filtered, scores_filtered = self.get_non_max_suppression_mask(data, top_scores, points, 75)
        points_to_return = self.points_to_response(points_filtered)
        plt.figure()
        plt.imshow(data)
        for x in range(len(points_filtered)):
            plt.text(points_filtered[x,1],points_filtered[x,0],str(x))
        plt.show()
        return points_to_return

    def get_non_max_suppression_mask(self, data: np.ndarray, scores: List, points: np.ndarray, threshold_radius: int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Suppress the inclusion of keypoints that fall within the same spatial area
        '''
        binary_image = np.zeros_like(data)
        # Find the largest responses and include these first
        response_list = scores
        mask = np.flip(np.argsort(response_list))
        point_list = points.astype(int)
        non_max_suppression_mask = []
        skipped = 0
        for point, index in zip(point_list, mask):
            # If the keypoint has no strong response neighbors already included
            if binary_image[point[0], point[1]] == 0:
                non_max_suppression_mask.append(index)
                # Update the binary immage so we can suppress future overlapping values
                cv.circle(binary_image, (point[0], point[1]), threshold_radius, 255, -1)
            else:
                skipped += 1
        print(non_max_suppression_mask)
        # Filter the keypoints using the good indicies
        return np.array(points)[non_max_suppression_mask], np.array(scores)[non_max_suppression_mask]

    def simple_scores_to_points(self, scores, data, num):
        points = []
        for i in range(num):
            # Get all the points that match this point
            p_dat = np.array(np.where(data == scores[i])).T
            print(p_dat)
            mean = np.mean(p_dat, axis=1)
            c_x = mean[0]
            c_y = mean[1]
            
            tPoint = Point()
            tPoint.x = c_x
            tPoint.y = c_y
            print(f"c_x: {c_x} c_y: {c_y} shape of curr img: {self.current_img.shape} shape of scores: {len(scores)}")
            tPoint.z = self.current_img[int(c_x),int(c_y)]
            points.append(tPoint)
            print(points)
        toSend = roiResponse()
        toSend.targets = points
        return toSend 

    def points_to_response(self, in_points):
        '''
        Calculate the centroid of n points
        '''
        points = []
        img = deepcopy(self.current_img)
        for point in in_points:
            try:
                print(point)
                tPoint = Point()
                # THIS IS A CORRECT SWAP
                tPoint.x = point[1]
                tPoint.y = point[0]
                tPoint.z = img[int(point[1]),int(point[0])]       
                points.append(tPoint)
            except Exception as e:
                print(e)
                print(f'Point {point} is bad with input of size {img.shape}')
        print(points)
        toSend = roiResponse()
        toSend.targets = points
        return toSend

    
    def scores_to_points(self, scores, data, num):
        '''
        Calculate the centroid of n points
        '''
        points = []
        for i in range(num):
            # Get all the points that match this point
            p_dat = np.array(np.where(data == scores[i])).T
            print(p_dat.shape)
            # Fit convex hull to points
            hull = ConvexHull(p_dat)
            # Find the centroid of this point
            vertices = hull.points[hull.vertices]
            c_x,c_y,_ = self.centroid_poly(vertices[:,0], vertices[:,1])
            if c_y > 1750 or c_y < 750:
                print(f"not a region of interest {c_x} {c_y}")
                continue      
            points.append([c_x,c_y])
        points = np.array(points)
        return points

    def centroid_poly(self, X, Y):
        """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon"""
        N = len(X)
        # minimal sanity check
        if not (N == len(Y)): raise ValueError('X and Y must be same length.')
        elif N < 3: raise ValueError('At least 3 vertices must be passed.')
        sum_A, sum_Cx, sum_Cy = 0, 0, 0
        last_iteration = N-1
        # from 0 to N-1
        for i in range(N):
            if i != last_iteration:
                shoelace = X[i]*Y[i+1] - X[i+1]*Y[i]
                sum_A  += shoelace
                sum_Cx += (X[i] + X[i+1]) * shoelace
                sum_Cy += (Y[i] + Y[i+1]) * shoelace
            else:
                # N-1 case (last iteration): substitute i+1 -> 0
                shoelace = X[i]*Y[0] - X[0]*Y[i]
                sum_A  += shoelace
                sum_Cx += (X[i] + X[0]) * shoelace
                sum_Cy += (Y[i] + Y[0]) * shoelace
        A  = 0.5 * sum_A
        factor = 1 / (6*A)
        Cx = factor * sum_Cx
        Cy = factor * sum_Cy
        # returning abs of A is the only difference to
        # the algo from above link
        return Cx, Cy, abs(A)

    def _image_callback(self, msg):
        self.current_img = self.br.imgmsg_to_cv2(msg)

if __name__=="__main__":
    ROI = RegionsOfInterest()
    rospy.spin()