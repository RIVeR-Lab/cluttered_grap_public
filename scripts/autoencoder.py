#!/usr/bin/env python3
import os
import rospy
import numpy as np
import rospkg

import numpy as np
from spectral import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             confusion_matrix, classification_report)
import rospkg
import cv2 as cv     
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from time import time
from numpy.random import seed
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from matplotlib import pyplot as plt
from cluttered_grasp.srv import autoencode, autoencodeResponse, autoencodereval, autoencoderevalResponse

# Authored by Nathaniel Hanson

class RealTimeAutoEncoder:

    def __init__(self):
        rospy.init_node("auto_encoder")
        rospy.loginfo(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
        # Run command to establish link to TensorFlow GPU libraries
        os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/river/miniconda3/lib/')
        self.pkg_path = rospkg.RosPack().get_path("cluttered_grasp")
        self.all_pixel_values = []
        # Grab Parameters from the rosparam server
        self.br = CvBridge()
        self.NUM_WAVE = rospy.get_param('~num_wavelengths', 273)
        self.loss_function = rospy.get_param('~loss_function', 'mse')
        self.encoding_dim = rospy.get_param('~encoding_dim', 20)
        self._create_network(self.encoding_dim, self.NUM_WAVE)
        self.dir = rospkg.RosPack().get_path("cluttered_grasp") # package path
        # self.autoencoder.load_weights(os.path.join(self.dir, 'hsi_autoencoder.h5'))
        # Create the training as a singleton function
        self.is_training = False
        self.dir = rospkg.RosPack().get_path("cluttered_grasp") # package path
        self.autoencoder.load_weights(os.path.join(self.dir, 'hsi_autoencoder.h5'))
        self.autoencode_evaluate = rospy.Service('/autoencoder/train', autoencode, self._call)
        self.autoencode_run = rospy.Service('/autoencoder/run', autoencode, self._run)
        self.reconstruction_service = rospy.Service('/autoencoder/reconstruct', autoencodereval, self._reconstruction_loss)
    
    def _create_network(self, encoding_dim: int, wave: int) -> None:
        # encoder dimension
        input_dim = Input(shape = (wave, ), name = 'InputLayer')
        # Encoder Layers
        encoded1 = Dense(150, activation = 'relu', name = 'EncodeLayer1')(input_dim)
        encoded2 = Dense(100, activation = 'relu', name = 'EncodeLayer2')(encoded1)
        encoded3 = Dense(75, activation = 'relu', name = 'EncodeLayer3')(encoded2)
        encoded4 = Dense(50, activation = 'relu', name = 'EncodeLayer4')(encoded3)
        # Latent Space Encoding
        encoded5 = Dense(encoding_dim, activation = 'linear', name = 'CodeLayer')(encoded4)
        # Decoder Layers
        decoded1 = Dense(50, activation = 'relu', name = 'DecodeLayer1')(encoded5)
        decoded2 = Dense(75, activation = 'relu', name = 'DecodeLayer2')(decoded1)
        decoded3 = Dense(100, activation = 'relu', name = 'DecodeLayer3')(decoded2)
        decoded4 = Dense(wave, activation = 'sigmoid', name = 'OutputLayer')(decoded3)
        # Combine Encoder and Deocder layers
        self.autoencoder = Model(inputs = input_dim, outputs = decoded4)

    def train(self, train_data: np.ndarray, loss_function: str = 'mse') -> None:
        self.is_training = True
        ### LOSS Functions
        # Spectral Information Divergence
        def spectral_information_divergence(y,y_reconstructed):
            t = tf.divide(tf.transpose(y),tf.reduce_sum(tf.transpose(y),axis=0))
            r = tf.divide(tf.transpose(y_reconstructed),tf.reduce_sum(tf.transpose(y_reconstructed),axis=0))
            loss = tf.reduce_sum(tf.reduce_sum(tf.multiply(t,tf.log(tf.divide(t,r))), axis=0)
                                + tf.reduce_sum(tf.multiply(r,tf.log(tf.divide(r,t)))), axis=0)
            return loss

        # Cosine of spectral angle loss
        def cos_spectral_angle_loss(y,y_reconstructed):
            normalize_r = tf.math.l2_normalize(tf.transpose(y_reconstructed),axis=0)
            normalize_t = tf.math.l2_normalize(tf.transpose(y),axis=0)
            loss = tf.reduce_sum(1-tf.reduce_sum(tf.multiply(normalize_r,normalize_t),axis=0))
            return loss
        
        # Set the loss function
        loss_all = {
            'mse':'mse',
            'cos': cos_spectral_angle_loss,
            'sid': spectral_information_divergence
        }

        # Compile the Model
        self.autoencoder.compile(optimizer = 'adam', 
                            loss = loss_all[loss_function], 
                            metrics = [tf.keras.metrics.MeanSquaredLogarithmicError()]
                            ) 

        ### Callbacks
        ## Early Stopping
        early_stop = EarlyStopping(monitor = 'loss',
                                    mode = 'min',
                                    min_delta = 0.000008,
                                    patience = 5,
                                    restore_best_weights = True)

        ## Checkpoint
        weights_path = os.path.join(self.dir, 'hsi_autoencoder.h5')
        checkpoint = ModelCheckpoint(filepath = weights_path, 
                                    monitor = 'mean_squared_logarithmic_error', 
                                    mode ='min', 
                                    save_best_only = True)

        ## Tensorboard
        tensorboard = TensorBoard(log_dir=self.dir + '/logs/{}'.format(time()))

        # Fit the Model
        hist = self.autoencoder.fit(train_data, 
                            train_data, 
                            epochs = 15, 
                            batch_size = 512, 
                            shuffle = True, 
                            callbacks=[early_stop,
                                        checkpoint,
                                        tensorboard])
        # Release the hold on model retraining
        self.is_training = False


    def predict(self, train_data: np.ndarray) -> np.ndarray:
        # Given a set of incoming pixels, classify them with the network
        prediction = np.zeros_like(train_data)
        # Chunk the data into blocks of 1000 pixels to pass through
        # To avoid overwhelming RAM
        chunks = np.arange(0,train_data.shape[0]+1000,1000)
        for i in range(len(chunks)-1):
            prediction[chunks[i]:chunks[i+1],:]= self.autoencoder(train_data[chunks[i]:chunks[i+1],:]).numpy()
        return prediction

    def _call(self, req):
        # Initialize response
        toSend = autoencodeResponse()
        if self.is_training:
            toSend.success = False
            return toSend
        print('Loaded all pixel values')
        self.all_pixel_values = np.load(self.pkg_path + "/associated_normalized_cube.npy")
        np.nan_to_num(self.all_pixel_values, copy=False, nan=0.0)
        print('Converting image mask')
        mask = self.br.imgmsg_to_cv2(req.mask, "passthrough").astype(np.bool)
        print(f'Mask dimensions: {mask.shape}')
        data = self.all_pixel_values[mask]
        print(f'Data dimensions: {data.shape}')
        # Otherwise, let's train the network
        # Grab pixel data and reshape it into a 2D array of pixels x wavelengths
        #input_data = req.data.reshape((req.n_pixel,req.n_wavelengths))
        self.train(data, req.loss_function)
        # Model is trained, return all pixels reconstructed
        #reconstruction = self.predict(data)
        toSend.data = [] #reconstruction.reshape((data.shape[0]*data.shape[1]))
        toSend.success = True
        return toSend

    def _reconstruction_loss(self, req: autoencodereval) -> autoencoderevalResponse:
        '''
        Given a set of pixels and a trained model, reconstruct them to detect anomalies
        '''
        toSend = autoencoderevalResponse()
        if self.is_training:
            toSend.success = False
        #     return toSend
        if self.all_pixel_values == []:
            print('Loading datacube!')
            self.all_pixel_values = np.load(self.pkg_path + "/associated_normalized_cube.npy")
            np.nan_to_num(self.all_pixel_values, copy=False, nan=0.0)
        print('Converting image to mask')
        mask = self.br.imgmsg_to_cv2(req.mask, "passthrough").astype(np.bool)
        print(f'Mask dimensions: {mask.shape}')
        input_data = self.all_pixel_values.reshape(self.all_pixel_values.shape[0]*self.all_pixel_values.shape[1],self.all_pixel_values.shape[2])
        print(f'Data dimensions: {input_data.shape}')    
        # Model is trained, return all pixels reconstructed
        reconstruction = self.predict(input_data)
        # Pass original data and reconstruction through the post lost function
        rec_loss = self.calculate_loss(input_data, reconstruction, req.loss_function)
        # Turn the loss values back into an image
        out_data = np.zeros_like(mask, dtype=np.float32)
        out_data[mask] = rec_loss
        plt.figure()
        plt.imshow(out_data)
        plt.show()
        np.save('/home/river/classifier_results.npy', out_data)
        toSend.result = self.br.cv2_to_imgmsg(out_data, "32FC1")
        toSend.success = True
        return toSend

    def calculate_loss(self, input_data: np.ndarray, reconstruction: np.ndarray, loss_function: str) -> np.ndarray:
        loss= np.zeros(input_data.shape[0])
        # Chunk the data into blocks of 1000 pixels to pass through
        # To avoid overwhelming RAM
        chunks = np.arange(0,input_data.shape[0]+1000,1000)
        for i in range(len(chunks)-1):
            if loss_function == 'mae':
                # Mean absolute error
                loss[chunks[i]:chunks[i+1]] = tf.keras.losses.mae(reconstruction[chunks[i]:chunks[i+1]], input_data[chunks[i]:chunks[i+1]]).numpy()
            elif loss_function == 'mse':
                # Mean square error
                loss[chunks[i]:chunks[i+1]] = tf.keras.losses.mse(reconstruction[chunks[i]:chunks[i+1]], input_data[chunks[i]:chunks[i+1]]).numpy()
            elif loss_function == 'sam':
                loss[chunks[i]:chunks[i+1]] = np.arccos((np.sum(reconstruction[chunks[i]:chunks[i+1]]*input_data[chunks[i]:chunks[i+1]],axis=1))/(np.linalg.norm(input_data[chunks[i]:chunks[i+1]], axis=1) * np.linalg.norm(reconstruction[chunks[i]:chunks[i+1]], axis=1)))
            else:
                rospy.logerr('Invalid reconstruction function!')
        return loss
    

    def _run(self, req: autoencode) -> autoencodeResponse:
        '''
        Pass incoming data through the autoencoder and return the reconstructed values
        '''
        toSend = autoencodeResponse()
        if self.is_training:
            toSend.success = False
            return toSend    
        # Grab pixel data and reshape it into a 2D array of pixels x wavelengths
        input_data = req.data.reshape((req.n_pixel,req.n_wavelengths))
        # Model is trained, return all pixels reconstructed
        # reconstruction = self.predict(input_data)
        toSend.data = [] #reconstruction.reshape((req.n_pixel*req.n_wavelengths))
        toSend.success = True
        return toSend

if __name__ == "__main__":
    AE = RealTimeAutoEncoder()
    rospy.spin()
