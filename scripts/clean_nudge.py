#!/usr/bin/env python3
from calendar import c
from lib2to3.pgen2.tokenize import TokenError
from anyio import start_blocking_portal
import numpy as np
import math
import threading
import rospy
import rospkg
import cv2 as cv
from typing import List
from matplotlib import pyplot as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from cluttered_grasp.srv import segment_plane, roi, roiResponse
from cluttered_grasp.srv import nudge, nudgeResponse
from spectral_finger_planner.srv import Move
from spectral_finger_planner.srv import Grasp
from geometry_msgs.msg import Point
from copy import deepcopy

# Authored by Gary Lvov and Nathaniel Hanson

class ClutterNudge:
    def __init__(self):
        rospy.init_node("revised_clutter_nudge")
        self.NUM_WAVELENGTHS = 273
        self.br = CvBridge()
        self.lock = threading.RLock()
        depth_image_sub = rospy.Subscriber('/depth_to_rgb/image_raw', Image, self._image_callback)
        self.pkg_path = rospkg.RosPack().get_path("cluttered_grasp")
        rospy.wait_for_service('ur3e_move_point')
        self.move_arm = rospy.ServiceProxy('ur3e_move_point', Move)
        rospy.wait_for_service('gripper_control')
        self.gripper_manip = rospy.ServiceProxy('gripper_control', Grasp)
        self.MAX_ANOMALIES = rospy.get_param('~max_anom', 10)
        self.HOME = [-0.25,-0.10,0.45]
        self.BBX = [0, 2000]
        self.BBY = [0, 2000] #TODO: choose dimensions of image
        self.PRIMITIVES_RADIUS_SIZE = 100
        self.PRIMITIVES_ANGLE_VAR = .314
        self.GRIPPER_OFFSET = 0.20
        self.GRIPPER_X_OFFSET = -0.09
        self.GRIPPER_Y_OFFSET = 0#-0.092
        self.CLUTTER_COL_TOL = 3 # millimeters
        self.CLUTTER_COL_WINDOW_SIZE = 50 #pixels
        self.CLUTTER_COL_STEP_SIZE = 1
        self.prep_move_arm(self.HOME)
        self.nudge_srv = rospy.Service("/nudge", nudge, self._call)
        print("Clutter Node Initialized succesfully")

    # Drives the whole clutter nudge pipleine 
    def _call(self, req):
        superpixels = self.br.imgmsg_to_cv2(req.superpixels)
        volumes = self.estimate_depth_cell_volume(self.current_img, superpixels)
        for point in req.targets:
            print(point)
            candidates = self.generate_candidate_plans(point)
            # TODO: Optimize our selection of the best candidate plan
            optimal_plan = []
            optimal_plan.append(candidates[0][0])
            optimal_plan.append(candidates[1][0])
            # optimal_plan = self.optimize_plans(candidates, superpixels, volumes) #TODO: Uncomment this
            optimal_plan[0] = self.convert_depth_pixel_to_metric_coordinate(pixel_x=optimal_plan[0][0], pixel_y=optimal_plan[0][1], 
            depth = optimal_plan[0][2])

            optimal_plan[1] = self.convert_depth_pixel_to_metric_coordinate(pixel_x=optimal_plan[1][0], pixel_y=optimal_plan[1][1], 
            depth=optimal_plan[1][2])
            
            coord = []
            coord.append(coord1)
            coord.append(coord2)
            # print(coord)
            # Need to maximize depth since that indicates furthest point to from the camera (lowest)
            min_depth = max(coord[0][2], coord[-1][2])
            coord[0][2] = min_depth
            coord[-1][2] = min_depth
            print(coord)
            success = self.execute_plan(coord)
        success = True
        return success

    # POINT HAS TO BE PROVIDED IN IMAGE SPACE
    def generate_candidate_plans(self, point: Point) -> np.ndarray:
            '''
            Given a point in R^3, heuristically generate candidate points that pass
            through the region

            returns: np.ndarray containing start and end points 
            [
                [
                    [x1,y1], [x1',y1']
                ],[
                    [x2,y2], [x2',y2']
                ],[
                    [x3,y3], [x3',y3']
                ]
            ]
            '''

            plt.gcf()

            potential_moves = []
            plt.plot(point.y, point.x, 'ro', color="pink", markersize=3)
            plt.text(point.y, point.x, "X's and Y's swapped")
            
            plt.plot(point.x, point.y, 'ro', color="purple", markersize=3)
            plt.text(point.x, point.y, "X's and Y's in positions provided")
            
            for angle in range(1, 11):
                x_offset = int(self.PRIMITIVES_RADIUS_SIZE * math.sin(angle * self.PRIMITIVES_ANGLE_VAR))
                y_offset = int(self.PRIMITIVES_RADIUS_SIZE * math.cos(angle * self.PRIMITIVES_ANGLE_VAR))
                # cast to int as pixels cannot be floats
                move = [[point.x - x_offset, point.y - y_offset, self.get_depth(point.x - x_offset, point.y - y_offset)], 
                        [point.x + x_offset, point.y + y_offset, self.get_depth(point.x + x_offset, point.y + y_offset)]]

                # Visualization has x and y's swapped
                plt.plot(point.x - x_offset, point.y - y_offset, 'ro', color="blue", markersize=1)
                plt.plot( point.x + x_offset, point.y + y_offset, 'ro', color="red", markersize=1)


                potential_moves.append(move)
                plt.draw()
            plt.show()
            return potential_moves

    def avoid_clutter_collisions(self, plan):
        #TODO: Actually write this function lol
        return plan

    def estimate_depth_cell_volume(self, depth, superpixels):
        '''
        Double integrate the xy values under the depth image to estimate
        the volume of the contained clutter
        '''
        plane_height = np.max(depth) - depth
        volumes = np.zeros_like(superpixels)
        for pixel in np.unique(superpixels):
            dtemp = deepcopy(plane_height)
            mask = superpixels == pixel
            dtemp[np.logical_not(mask)] = 0.00
            # Double integrate to get the volume
            volumes[mask] = np.trapz(np.trapz(dtemp,axis=0))
        return volumes

    # Convert plans to real-space ---------------------------------------------------------------------
    def convert_plan(self, plan):
        newPlan = []
        depth = max(self.current_img[int(plan[0][0]), int(plan[0][1])], self.current_img[int(plan[1][0]), int(plan[1][1])])

        for x in range(2):
            newPlan.append(self.convert_depth_pixel_to_metric_coordinate(depth, plan[x][0], plan[x][1]))

        return newPlan

    def convert_depth_pixel_to_metric_coordinate(self, pixel_x=0, pixel_y=0, depth=0):
        """
        Convert the depth and image point information to metric coordinates
        Parameters:
        -----------
        depth 	 	 	 : double
                                The depth value of the image point
        pixel_x 	  	 	 : double
                                The x value of the image coordinate
        pixel_y 	  	 	 : double
                                The y value of the image coordinate
        Return:
        ----------
        X : double
            The x value in meters
        Y : double
            The y value in meters
        Z : double
            The z value in meters
        """
        X = (pixel_x - 1028.62)/972.764 *depth/1000
        Y = (pixel_y - 776.698)/972.484 *depth/1000
        return [X, Y, depth/1000]

    

    # Arm Movement ------------------------------------------------------------
    def prep_move_arm(self, position: List) -> bool:
        '''
        Move arm to requested x,y,z point
        # '''
        toSendMeta = Move()
        toSend = Point()
        toSend.x = np.round(position[0],4)
        toSend.y = np.round(position[1],4)
        toSend.z = np.round(position[2],4)
        toSendMeta.request = toSend
        response = self.move_arm(toSend)
        return response

    def execute_plan(self, plan: np.ndarray) -> bool:
        '''
        Move the arm from the start/end position of the gripper using PyKDL and the move point service
        '''
        # Move arm to start position
        self.gripper_manip('close')
        # TAKE HOME POSITION
        for x in range(5):
            newTargetZ = np.clip(self.HOME[2] - plan[0][2] + self.GRIPPER_OFFSET,-0.5,0.5)
            # Calculate flipped x and y positions        self.HOME = [-0.25,-0.10,0.45]
            newTargetX = np.clip(plan[0][1] + self.HOME[0] + self.GRIPPER_X_OFFSET,-0.45,-0.1)
            newTargetY = np.clip(plan[0][0] + self.HOME[1] + self.GRIPPER_Y_OFFSET,-0.45,0.45)
            print(f'Moving arm to: {[newTargetX, newTargetY, newTargetZ+0.1]}')
            #elf.prep_move_arm([newTargetX, newTargetY, newTargetZ+0.1])

        # Lower arm into positions
        self.prep_move_arm([newTargetX, newTargetY, newTargetZ])
        # Move across surface area
        for _ in range(5):
            newTargetX = np.clip(plan[1][1] + self.HOME[0] + self.GRIPPER_X_OFFSET,-0.4,-0.1)
            newTargetY = np.clip(plan[1][0] + self.HOME[1] + self.GRIPPER_Y_OFFSET,-0.4,0.4)
            newTargetZ = np.clip(self.HOME[2] - plan[1][2] + self.GRIPPER_OFFSET,-0.5,0.5)
            print(f'Moving arm to: {[newTargetX, newTargetY, newTargetZ+0.1]}')
            #elf.prep_move_arm([newTargetX, newTargetY, newTargetZ])

        # Move arm to ready position
        for _ in range(5):
            # Return to the original position
            newTargetZ = np.clip(self.HOME[2] - plan[0][2] + self.GRIPPER_OFFSET,-0.5,0.5)
            # Calculate flipped x and y positions        self.HOME = [-0.25,-0.10,0.45]
            newTargetX = np.clip(plan[0][1] + self.HOME[0] + self.GRIPPER_X_OFFSET,-0.4,-0.1)
            newTargetY = np.clip(plan[0][0] + self.HOME[1] + self.GRIPPER_Y_OFFSET,-0.4,0.4)
            #self.prep_move_arm([newTargetX, newTargetY, newTargetZ])
            print(f'Moving arm to: {[newTargetX, newTargetY, newTargetZ+0.1]}')

        self.prep_move_arm([newTargetX, newTargetY, newTargetZ+0.1]) #TODO: Not sure what this line does, mayb remove
        # Move arm home
        self.prep_move_arm(self.HOME)
        self.gripper_manip('open')

    # callbacks and utilities ------------------------------------------------
    def _image_callback(self, msg):
        with self.lock:
            self.current_img = self.br.imgmsg_to_cv2(msg)
            
    def get_depth(self,x,y):
        ### This needs the x and y to be flipped
        y,x = x,y
        with self.lock:
            print(self.current_img.shape)
            return self.current_img[x,y]