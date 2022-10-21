#!/usr/bin/env python3

from asyncore import file_dispatcher
import time
import sys
import os
from turtle import pos, pu
import rospy
import typing
import traceback
from datetime import datetime
from typing import List, Dict, Tuple
from std_msgs.msg import String, Header
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt
import copy
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import rospkg
from headwall_ros.msg import Cube
from robo_rail.srv import roboRail
from std_srvs.srv import Trigger, TriggerResponse

from cv_bridge import CvBridge
from cluttered_grasp.srv import Rescan, RescanResponse, nudge, segment_plane

from headwall_ros.srv import CubeRequest, CubeCommand, CubeRequestResponse, CubeCommandResponse, CubeSave, CubeSaveResponse
from spectral_finger_planner.srv import Move, MoveResponse
from cluttered_grasp.srv import several, superpixel, pure_kmeans, autoencode, autoencodereval, roi

# collect_hypercube.py
# Author: Nathaniel Hanson and Gary Lvov
# Meta node to manage collection of hyperspectral data cube use HSI rail-robot workcell

class HypercubeCollector():
    def __init__(self):
        ### Defined parameters of Headwall Nano
        self.CHANNELS = 273
        self.COLUMNS = 640
        self.MIDDLE_RAIL = 11500
        self.START_RAIL = 28500
        self.END_RAIL = 0
        self.positions = [
            [-0.40, -0.10, 0.40],
            [-0.30, -0.10, 0.40],
            [-0.20, -0.10, 0.40]
        ]
        self.picture_pose = [-0.25,-0.10,0.45]
        self.move_service = rospy.Service('rescan_surface', Rescan, self.rescan)
        self.br = CvBridge()

        self.num_scan = 0
        self.pkg_path = rospkg.RosPack().get_path("cluttered_grasp") # package path
        # Initialize services for hypercube processing
        rospy.wait_for_service('start_cube')
        self.start = rospy.ServiceProxy('start_cube', CubeCommand)
        rospy.wait_for_service('pause_cube')
        self.pause = rospy.ServiceProxy('pause_cube', CubeCommand)
        rospy.wait_for_service('clear_cube')
        self.clear = rospy.ServiceProxy('clear_cube', CubeCommand)
        rospy.wait_for_service('save_cube')
        self.save = rospy.ServiceProxy('save_cube', CubeSave)
        rospy.wait_for_service('ur3e_move_point')
        self.move_arm = rospy.ServiceProxy('ur3e_move_point', Move)
        # Intialize services for rail control
        rospy.wait_for_service('robo_rail_listener')
        self.railCommand = rospy.ServiceProxy('robo_rail_listener', roboRail)
        # Initialize ROI services
        rospy.wait_for_service('/roi')
        self.roi = rospy.ServiceProxy('/roi', roi)
        # Intialize autoencoder service
        rospy.wait_for_service('/autoencoder/train')
        self.autoTrain = rospy.ServiceProxy('/autoencoder/train', autoencode)
        rospy.wait_for_service('/autoencoder/reconstruct')
        self.autoEval = rospy.ServiceProxy('/autoencoder/reconstruct', autoencodereval)

        self.rescan_full_srv = rospy.Service("/rescan_full", Trigger, self._rescan_full)

    def move_rail(self, pos: int, vel: int) -> str:
        '''
        Move the rail to a specificed position at the correct velocity
        '''
        msg = self.railCommand("addToPosQueue", pos, vel)
        msg = self.railCommand("runQueue", 0, 0)
        return msg

    def prep_move_arm(self, position: List) -> bool:
        '''
        Move arm to requested x,y,z point
        '''
        toSendMeta = Move()
        toSend = Point()
        toSend.x = position[0]
        toSend.y = position[1]
        toSend.z = position[2]
        toSendMeta.request = toSend
        response = self.move_arm(toSend)
        return response

    def main(self) -> None:
        '''
        End to end hypercube collection
        # '''
        # filePaths = ["/home/river/results/limes/initial_cubes/l1.npy", 
        #             "/home/river/results/limes/initial_cubes/l2.npy", 
        #             "/home/river/results/limes/initial_cubes/l3.npy"]
        
        if(self.num_scan > 0):
            if(os.path.exists(self.pkg_path + "/associated_normalized_cube.npy")):
                os.rename(self.pkg_path + "/associated_normalized_cube.npy",
                        self.pkg_path + "/associated_normalized_cube" + str(self.num_scan) + ".npy")
        filePaths = self.run_collect()
        
        data = self.fuse_data(filePaths)
        
        # '''
        # First, stitches together all HSI into one hsi

        # Compares fused HSI to RGB data, obtaining the homography

        # Transform the hyperspectral cubes according to previously obtained homography,
        # creating a 1-1 association between pixels in the RGB image and pixels in the 
        # hyperspectral cube.
        # '''

        print("Finding Superpixels!")
        superpixels = self.get_superpixels()
         
        superpix_visu = self.br.imgmsg_to_cv2(superpixels, "passthrough")
        # path = "/home/river/cluttered_ws/src/cluttered_grasp/images/vest"
        # cv2.imwrite(os.path.join(path , "superpixels.png"), superpix_visu)

        # pure_img = self.get_pure_regions(superpixels)
        # mask = self.br.cv2_to_imgmsg(pure_img, encoding="passthrough")
        # print('Done!')

        # print("Using a fancy autoencoder...")
        # if self.num_scan == 0:
        #     print("training...")
        #     train = self.autoTrain(mask, self.CHANNELS, 'mse')
            
        print("Loading...")
        mask = self.br.cv2_to_imgmsg(np.ones(superpix_visu.shape), encoding="passthrough")
        
        loss = self.autoEval(mask, 'mae')


        print("determining regions of interest...")
        try:
            points = self.roi(superpixels, loss.result).targets
            
        except Exception as e:
            print(traceback.print_exc())
            print("ROI messed up. continuing regardless")

        print(f"Nudging : Roi: {points}")
        nudge = self.nudge_objects(points, superpixels)

        self.num_scan += 1
        print(f"Number of scans performed: {self.num_scan}")

    def get_plane(self):
        rospy.wait_for_service("/segment_plane")
        try:
            get_seg = rospy.ServiceProxy('/segment_plane', segment_plane)
            seg_img = get_seg().image
            # seg_img = self.br.imgmsg_to_cv2(seg_img)
            return seg_img

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def run_collect(self):
        filePaths = []

        for position in self.positions:
            # Move rail to middle        print(img.shape)
            msg = self.move_rail(self.MIDDLE_RAIL, 50)
            # 
            print(msg)
            print('Complete!')
            # Move arm to position
            msg = self.prep_move_arm(position)
            print(msg)
            # Move rail to start
            msg = self.move_rail(self.START_RAIL, 50)
            
            print(msg)
            print('Rail ready for cube!')
            # Start cube collection
            msg = self.start('Start')
            print(msg)
            print('Cube collection started!')
            # Move arm along length of workspace
            msg = self.move_rail(self.END_RAIL, 2)
            
            print('Rail done!')
            print(msg)
            # Stop hypercube collection
            msg = self.pause('Pause')
            print(msg)
            # Move rail to middle to prevent lights from overheating it
            msg = self.move_rail(self.MIDDLE_RAIL, 50)
            
            print(msg)
            # Save cube and report location
            msg = self.save('/home/river/datacubes', '')
            print(msg)
            # Keep track of all the hypercubes we are collections
            filePaths.append(msg.filepath)
            # Clear cube to prevent excess data acculumation in memory
            msg = self.clear('Clear')
            print(msg)

        return filePaths

    def fuse_data(self, filepaths):
        rospy.wait_for_service('/associate')
        try:
            rgb_reg = rospy.ServiceProxy('/associate', several)
            resp1 = rgb_reg(filepaths)
            return resp1
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def get_superpixels(self):
        rospy.wait_for_service("/superpixel")
        try:
            get_image = rospy.ServiceProxy("/superpixel", superpixel)
            img = get_image().image # get the image in bytes format
            return img
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def get_pure_regions(self, superpixels):
        rospy.wait_for_service("/pure_with_kmeans")
        try:
            get_pure = rospy.ServiceProxy('/pure_with_kmeans', pure_kmeans)
            print('Applying kmeans')
            pure_img = get_pure(superpixels).pure_regions
            pure_img = self.br.imgmsg_to_cv2(pure_img)
            return pure_img
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def nudge_objects(self, targets, superpixels):
        rospy.wait_for_service("/nudge")
        try:
            get_nudge = rospy.ServiceProxy('/nudge', nudge)
            nudge_res_bool = get_nudge(targets, superpixels).res
            return nudge_res_bool
            
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def _rescan_full(self, req):
        self.main()
        return TriggerResponse()
    
    # def run_recollect(self):
    #     filePaths = []

    #     for position in self.positions:
    #     # Move rail to middle        print(img.shape)
    #         # Move arm to position
    #         msg = self.prep_move_arm(position)
    #         print(msg)
    #         # Move rail to start
    #         msg = self.move_rail(self.START_RAIL, 50)
    #         print(msg)
    #         print('Rail ready for cube!')
    #         # Start cube collection
    #         msg = self.start('Start')
    #         print(msg)
    #         print('Cube collection started!')
    #         # Move arm along length of workspace
    #         msg = self.move_rail(self.END_RAIL, 50)
    #         print('Rail done!')
    #         print(msg)
    #         # Stop hypercube collection
    #         msg = self.pause('Pause')
    #         print(msg)
    #         msg = self.save('/home/river/datacubes', '')
    #         print(msg)
    #         # Keep track of all the hypercubes we are collections
    #         filePaths.append(msg.filepath)
    #         # Clear cube to prevent excess data acculumation in memory
    #         msg = self.clear('Clear')
    #         print(msg)

    #     # return filePaths

    def rescan(self, msg: Rescan) -> RescanResponse:
        '''
        Rescan a small portion of the table surface
        '''
        # Move arm to correct position
        _ = self.prep_move_arm([msg.arm_point.x, msg.arm_point.y, msg.arm_point.z])
        # Move rail to start
        _ = self.move_rail(self.START_RAIL, 50)
        # Start hypercube collection
        _ = self.start('Start')
        # Start translation
        _ = self.move_rail(self.END_RAIL, 2)
        # Stop hypercube collection
        _ = self.pause('Pause')
        # Move rail to middle to prevent lights from overheating it
        _ = self.move_rail(self.MIDDLE_RAIL, 50)
        # Save cube and report location
        saveMsg = self.save('/home/river', '')
        # Extract the file path
        filepath = saveMsg.filepath
        # Clear cube to prevent excess data acculumation in memory
        _ = self.clear('Clear')
        # Give the updated path back for the user to process
        toSend = RescanResponse()
        toSend.filepath = filepath
        return filepath
    
    def shutdown(self) -> None:
        '''
        Saves the datacube cube to a temp directory before gracefully exiting
        '''
        pass

# Main functionality
if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('hypercube_collect', anonymous=True)
    try:
        collector = HypercubeCollector()
        collector.main()
        rospy.spin()
    except rospy.ROSInterruptException:
        collector.shutdown()

#  def main(self) -> None:
#         '''
#         End to end hypercube collection
#         '''
#         # filePaths = ["/home/river/datacubes/block1.npy", "/home/river/datacubes/block2.npy", "/home/river/datacubes/block3.npy"]
        
#         if(self.num_scan > 0):
#             if(os.path.exists(self.pkg_path + "/associated_normalized_cube.npy")):
#                 os.rename(self.pkg_path + "/associated_normalized_cube.npy",
#                         self.pkg_path + "/associated_normalized_cube" + str(self.num_scan) + ".npy")
#         filePaths = self.run_collect()
        
#         data = self.fuse_data(filePaths)
        
#         # '''
#         # First, stitches together all HSI into one hsi

#         # Compares fused HSI to RGB data, obtaining the homography

#         # Transform the hyperspectral cubes according to previously obtained homography,
#         # creating a 1-1 association between pixels in the RGB image and pixels in the 
#         # hyperspectral cube.
#         # '''

#         # print("Finding Superpixels!")
#         superpixels = self.get_superpixels()
         
#         superpix_visu = self.br.imgmsg_to_cv2(superpixels, "passthrough")
#         path = "/home/river/cluttered_ws/src/cluttered_grasp/images/vest"
#         cv2.imwrite(os.path.join(path , "superpixels.png"), superpix_visu)

#         pure_img = self.get_pure_regions(superpixels)
#         mask = self.br.cv2_to_imgmsg(pure_img, encoding="passthrough")
#         print('Done!')

        
#         # print("Using a fancy autoencoder...")
#         if self.num_scan == 0:
#             print("training...")
#             train = self.autoTrain(mask, self.CHANNELS, 'mse')
            

#         print("Loading...")
#         mask = self.br.cv2_to_imgmsg(np.ones(superpix_visu.shape), encoding="passthrough")
        
#         seg_mask = self.get_plane()
#         # mask = seg_mask
#         seg_mask_visu = self.br.imgmsg_to_cv2(seg_mask, "passthrough")
#         plt.figure()
#         plt.imshow(seg_mask_visu)
#         plt.show()
#         loss = self.autoEval(mask, 'mae')
#         img = loss.result
#         img = self.br.imgmsg_to_cv2(img).reshape(seg_mask_visu.shape)
#         print(img.shape)
#         print(seg_mask_visu.shape)
        
#         img_cp = img.copy()
        
#         img_cp[(seg_mask_visu == 1)] = 0

#         plt.figure()
#         plt.imshow(img_cp)
#         plt.show()
        
#         img_cp = self.br.cv2_to_imgmsg(img_cp)

#         print("determining regions of interest...")
#         try:
#             points = self.roi(superpixels, loss.result).targets
            
#         except rospy.service.ServiceException as e:
#             print("ROI messed up. continuing regardless")

#         print(f"Nudging : Roi: {points}")
#         nudge = self.nudge_objects(points, superpixels)

#         self.num_scan += 1
#         print(f"Number of scans performed: {self.num_scan}")
