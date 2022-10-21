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
from typing import List, Tuple
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
        rospy.init_node("cn")
        # Define the max number of wavelengths currently available
        self.NUM_WAVELENGTHS = 273
        # Initiate bridge to convert ROS messages to numpy arrays
        self.br = CvBridge()
        # Create a lock to protect access to current depth image
        self.lock = threading.RLock()
        # Subscribe to average depth image
        depth_image_sub = rospy.Subscriber('/depth_to_rgb/image_raw', Image, self._image_callback)
        self.pkg_path = rospkg.RosPack().get_path("cluttered_grasp")
        # Wait for UR3e control service to be available
        rospy.wait_for_service('ur3e_move_point')
        self.move_arm = rospy.ServiceProxy('ur3e_move_point', Move)
        # Wait for 2F-85 gripper to be available
        rospy.wait_for_service('gripper_control')
        self.gripper_manip = rospy.ServiceProxy('gripper_control', Grasp)
        # Limit the number of points we are going to perturb in the image scene
        self.MAX_ANOMALIES = rospy.get_param('~max_anom', 10)
        # This defines the home pose of the UR3 arm in the base_link frame
        self.HOME = [-0.25,-0.10,0.45]
        # Define the size of the initial cadidate radius to be used
        self.PRIMITIVES_RADIUS_SIZE = 100
        # Offset of space between cadidate lines
        self.PRIMITIVES_ANGLE_VAR = np.pi/10
        # These variables define hard coded offsets considering the size of the gripper
        # and some good ol' experimentally defined constants
        self.GRIPPER_OFFSET = 0.20
        self.GRIPPER_X_OFFSET = -0.09
        self.GRIPPER_Y_OFFSET = 0
        # These parameters define the steps and window to use while optimizing the pushing motion plan
        self.CLUTTER_COL_TOL = 3 # millimeters
        self.CLUTTER_COL_WINDOW_SIZE = 50 #pixels
        self.CLUTTER_COL_STEP_SIZE = 1
        # Move the arm to the home position if it is not already there
        self.prep_move_arm(self.HOME)
        self.nudge_srv = rospy.Service("/nudge", nudge, self._call)
        print("Node Initialized succesfully")

    def _call(self, req):
        superpixels = self.br.imgmsg_to_cv2(req.superpixels)
        volumes = self.estimate_depth_cell_volume(self.current_img, superpixels)
        plt.figure()
        # ONLY RUN THE BEST THREE POINTS
        for point in req.targets[:3]:
            print(point)
            candidates = self.generate_candidate_plans(point)
            # Extend points to avoid impacting clutter
            new_plans = [self.extend_plan(plan) for plan in candidates]
            # Filter out all plans that are none
            new_plans_filtered = [plan for plan in new_plans if plan != None]
            # Optimize plans to maximize impact with clutter
            optimal_plan = self.optimize_plans(new_plans_filtered, superpixels, volumes)
            # Ensure start point of trajectory always starts on the plane
            new_plan = self.swap_start(optimal_plan)
            # Constrain both points to be at the maximum depth
            new_plan = self.maximize_depth(new_plan)
            start, end = new_plan
            print(f"start: {start} end: {end}")
            plt.imshow(self.current_img)
            plt.plot(start[0], start[1], 'ro', color="orange", markersize=5)
            plt.plot(end[0], end[1], 'ro', color="purple", markersize=5)
            # Convert from pixel coordinates to depth coordinates
            start_coord = self.convert_depth_pixel_to_metric_coordinate(pixel_x=start[0], pixel_y=start[1], depth=start[2])
            end_coord = self.convert_depth_pixel_to_metric_coordinate(pixel_x=end[0], pixel_y=end[1], depth=end[2])
            coord = [start_coord, end_coord]
            print(coord)
            plt.show()
            success = self.execute_plan(coord)
        success = True
        return success

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

    # POINT HAS TO BE PROVIDED IN IMAGE SPACE
    def generate_candidate_plans(self, point: Point) -> np.ndarray:
        img = self.current_img
        START_ANGLE = np.pi/4
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
        potential_moves = []
        plt.imshow(img)
        plt.plot(point.y, point.x, 'ro', color="pink", markersize=3)
        plt.text(point.y, point.x, "X's and Y's swapped")
        
        plt.plot(point.x, point.y, 'ro', color="purple", markersize=3)
        plt.text(point.x, point.y, "X's and Y's in positions provided")
        
        for angle in range(2,10):
            x_offset = int(self.PRIMITIVES_RADIUS_SIZE * math.sin(angle * self.PRIMITIVES_ANGLE_VAR))
            y_offset = int(self.PRIMITIVES_RADIUS_SIZE * math.cos(angle * self.PRIMITIVES_ANGLE_VAR))
            # cast to int as pixels cannot be floats
            move = [[point.x - x_offset, point.y - y_offset, self.get_depth(int(point.x - x_offset), int(point.y - y_offset))], 
                    [point.x + x_offset, point.y + y_offset, self.get_depth(int(point.x + x_offset), int(point.y + y_offset))]]

            # Visualization has x and y's swapped
            plt.plot(point.x - x_offset, point.y - y_offset, 'ro', color="blue", markersize=1)
            plt.plot( point.x + x_offset, point.y + y_offset, 'ro', color="red", markersize=1)

            potential_moves.append(move)
            plt.draw()
        plt.show()
        return potential_moves
    

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
            self.prep_move_arm([newTargetX, newTargetY, newTargetZ+0.1])

        # Lower arm into positions
        self.prep_move_arm([newTargetX, newTargetY, newTargetZ])
        # Move across surface area
        for _ in range(2):
            newTargetX = np.clip(plan[1][1] + self.HOME[0] + self.GRIPPER_X_OFFSET,-0.4,-0.1)
            newTargetY = np.clip(plan[1][0] + self.HOME[1] + self.GRIPPER_Y_OFFSET,-0.4,0.4)
            newTargetZ = np.clip(self.HOME[2] - plan[1][2] + self.GRIPPER_OFFSET,-0.5,0.5)
            print(f'Moving arm to: {[newTargetX, newTargetY, newTargetZ+0.1]}')
            self.prep_move_arm([newTargetX, newTargetY, newTargetZ])

        # Move arm to ready position
        for _ in range(2):
            # Return to the original position
            newTargetZ = np.clip(self.HOME[2] - plan[0][2] + self.GRIPPER_OFFSET,-0.5,0.5)
            # Calculate flipped x and y positions        self.HOME = [-0.25,-0.10,0.45]
            newTargetX = np.clip(plan[0][1] + self.HOME[0] + self.GRIPPER_X_OFFSET,-0.4,-0.1)
            newTargetY = np.clip(plan[0][0] + self.HOME[1] + self.GRIPPER_Y_OFFSET,-0.4,0.4)
            self.prep_move_arm([newTargetX, newTargetY, newTargetZ])
            print(f'Moving arm to: {[newTargetX, newTargetY, newTargetZ+0.1]}')

        self.prep_move_arm([newTargetX, newTargetY, newTargetZ+0.1])
        # Move arm home
        self.prep_move_arm(self.HOME)
        self.gripper_manip('open')

    def _image_callback(self, msg):
        with self.lock:
            self.current_img = self.br.imgmsg_to_cv2(msg)
            
    def get_depth(self,x,y,img=[]):
        ### This needs the x and y to be flipped
        y,x = x,y
        #with self.lock:
        if len(img) > 0:
            return img[x,y]
        return self.current_img[x,y]

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
        
    def find_intersection_cells(self, x, y, superpixels):
        '''
        Given a line with points x,y estimate which super pixels the line
        uniquely crosses through
        '''
        coords = np.vstack((x,y)).T
        print(superpixels.shape)
        print(coords)
        dat = []
        for coord in coords:
            print(f'Coordinate: {coord}')
            x,y = coord
            tVal = superpixels[x,y]
            print(f'Value: {tVal}')
            dat.append(tVal)

        return np.unique(dat)

    def find_tot_volume(self, cells: np.ndarray, superpixels: np.ndarray, volume: np.ndarray) -> float:
        '''
        Given a set of superpixel values, sum up total occupied volume
        '''
        cum_volume = 0
        for cell in cells:
            cum_volume += np.median(volume[superpixels == cell])
        return cum_volume

    def swap_start(self, plan: np.ndarray) -> np.ndarray:
        '''
        Swap the start and end of a plan to make sure the start point is the one closest to the plane
        '''
        return (plan[1], plan[0]) if plan[1][2] > plan[0][2] else plan

    def maximize_depth(self, plan: np.ndarray) -> np.ndarray:
        '''
        Constrain movements to the maximum z coordinate to prevent diagonal motions over the clutter
        '''
        use_depth = max(plan[0][2], plan[0][2])
        return [[*plan[0][0:2], use_depth], [*plan[1][0:2], use_depth]]

    def optimize_plans(self, plans: np.ndarray, superpixels: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        '''
        Given a number of candidate plans by measuring the volume of impacted superpixels along 
        the selected cartesian trajectory

        return: np.ndarray consisting of start/end positions of the target plan
        '''
        scores = []
        for plan in plans:
            print(plan)
            xs = np.linspace(plan[0][1], plan[1][1], num=100, dtype=int)
            ys = np.linspace(plan[0][0], plan[1][0], num=100, dtype=int)
            cells = self.find_intersection_cells(xs,ys,superpixels)
            scores.append(self.find_tot_volume(cells, superpixels, volumes))
        print(f'Target plan selected: {np.argmin(scores)}, {plans[np.argmin(scores)]}')
        return plans[np.argmax(scores)]

    def extend_plan(self, plan):
        # plt.figure()
        # plt.imshow(self.current_img)
        STEP_SIZE = 1
        start, end = plan
        divisor = (start[0] - end[0]) or 0.1
        slope = (start[1] - end[1]) / divisor
        print(f'Slope: {slope}')
        # new values are either tuple with four elements, or none
        # four elements of tuple are x, y, z, and steps taken until safe
        new_start, new_end = self.extend_point(start, STEP_SIZE * -1, slope), self.extend_point(end, STEP_SIZE, slope)
        print(f'New Start: {new_start}')
        print(f'New End: {new_end}')
        if new_start is not None and new_end is not None:
            # Select an entry point based off of which one is closer: 
            if(new_start[3] > new_end[3]):
                return [list(new_start[:3]), end]
            else:
                return [start, list(new_end[:3])]
        
        elif new_start is not None and new_end is None:
            return [list(new_start[:3]), end]
        
        elif new_start is None and new_end is not None:
            return [start, list(new_end[:3])]
        else:
            return None

    def extend_point(self, point: List[int], step_size: int, slope: float) -> Tuple[int] or None:
        PLANE_HEIGHT = 570
        PLANE_TOL = 15 # millimeters
        BAD_TOL = 25 # number of pixels encountered that do not meet window criteria before reset
        WINDOW_TOL = 50 #number of good pixels encountered before deeming success
        Y_MIN = 100
        Y_MAX = 950
        
        steps_taken = 0
        bad_count = 0
        
        window = []
        new_pt = point
        current_img = deepcopy(self.current_img)
        while Y_MIN <= new_pt[1] <= Y_MAX:
            new_pt = [int(point[0] + (step_size * steps_taken)), 
                    int(point[1] + (step_size * steps_taken * slope)), 
                    self.get_depth(x=int(point[0] + step_size * steps_taken),
                                y=int(point[1] + step_size * steps_taken * slope), img=current_img)]
            
            
            
            if(new_pt[2] - PLANE_TOL <= PLANE_HEIGHT <= new_pt[2] + PLANE_TOL):
                window.append(new_pt)
                # plt.plot(new_pt[0], new_pt[1], 'ro', color="blue", markersize=1)
            else:
                bad_count += 1
                # plt.plot(new_pt[0], new_pt[1], 'ro', color="red", markersize=1)
                
            if(bad_count >= BAD_TOL):
                window = []
                bad_count = 0
                
            elif(len(window) >= WINDOW_TOL):
                print('Criteria satisfied')
                return *new_pt, steps_taken
            
            steps_taken += 1
        # plt.draw()
        # plt.show(block=True)
        return None

if __name__=="__main__":
    nudger = ClutterNudge()
    rospy.spin()