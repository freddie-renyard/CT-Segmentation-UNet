import os
import sys
import random
from PIL import Image
from matplotlib import cm
import math
import numpy as np
from datetime import datetime

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input, Reshape, Multiply
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataGen(keras.utils.Sequence):

    # The elastic_transform function below was sourced from the Kaggle notebook here 
    # (https://www.kaggle.com/babbler/mnist-data-augmentation-with-elastic-distortion), 
    # with code written by 'Joe G'
    
    def elastic_transform(self, image, alpha_range, sigma, random_state=None):
    
        '''
        Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        
    # Arguments
        image: Numpy array with shape (height, width, channels). 
        alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
            Controls intensity of deformation.
        sigma: Float, sigma of gaussian filter that smooths the displacement fields.
        random_state: `numpy.random.RandomState` object for generating displacement fields.
        
        '''
        
        if random_state is None:
            random_state = np.random.RandomState(None)
            
        if np.isscalar(alpha_range):
            alpha = alpha_range
        else:
            alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    # Settings for the data generators that generate training and testing data for the model

    def __init__(self, batch_size=1, image_size=256):

        self.path_train = '/path/to/training/directory'
        self.path_test = '/path/to/test/directory'
        self.batch_size = batch_size
        self.image_size = image_size

        classes = ['epidural', 'intraparenchymal', 'intraventricular', 'none', 'subarachnoid', 'subdural']

        # The data generator for the training data, which includes all of the data augmentation
        data_gen = ImageDataGenerator(
            rescale=1.0/255.0,
            horizontal_flip=True, # Flips the images about the x axis
            rotation_range=40, # Random rotation
            shear_range=4, # Random shear to imitate viewing the image on a screen from various angles
            zoom_range=[0.8, 1.5], # Random zooming
            brightness_range=[0.3, 1.8], # Random brightness to reflect different windows on imaging
            preprocessing_function=lambda x: self.elastic_transform(x, alpha_range=[0,20], sigma=3) 
            # Elastic deformation to reflect tissue morphology differences between patients
        )


        # The data generator for the testing data, which includes no data augmentation
        normal_data_gen = ImageDataGenerator(
            rescale = 1.0/255.0
        )

        self.train = data_gen.flow_from_directory(
            (self.path_train),
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            classes=classes,
            color_mode='rgb',
            class_mode='categorical',
            shuffle=True
        )

        self.test = normal_data_gen.flow_from_directory(
            (self.path_test),
            target_size=(self.image_size, self.image_size),
            batch_size=65,
            classes=classes,
            color_mode='rgb',
            class_mode='categorical',
            shuffle=True
        )

    # The functions for loading images and splitting them from their RGB format into
    # images (x) and masks (y)
    
    def load_training_data(self):

        img = self.train.next()[0]
        
        x = img[:, :, :, 0]
        y = img[:, :, :, 1]
        x = np.expand_dims(x, axis=3)
        y = np.expand_dims(y, axis=3)

        return x, y

    def load_testing_data(self):

        img = self.test.next()[0]
        x = img[:, :, :, 0]
        y = img[:, :, :, 1]
        x = np.expand_dims(x, axis=3)
        y = np.expand_dims(y, axis=3)

        return x, y


# Functions for plotting predictions of the model using Pyplot

class Utils:

    # Displays predictions of the model passed to it in the form of the mask layed 
    # over the original scan
    
    def display_heatmap(model, no_of_predictions=1):

        data = DataGen()
        for j in range(no_of_predictions):
            x,y = data.load_testing_data()
            predictions = model.predict_on_batch(x)
            Utils.heatmap(scan=x[0], prediction=predictions[0], ground_truth=y[0])

    def heatmap(scan, prediction, ground_truth):
        
        resolution = scan.shape[0]
        blue = np.zeros((resolution,resolution,1))
        img = np.concatenate((scan, prediction, blue), axis=2)

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.title('Haemorrhage Prediction')
        plt.imshow(img)

        img_2 = np.concatenate((scan, ground_truth, blue), axis=2)

        plt.subplot(1,2,2)
        plt.axis('off')
        plt.title('Ground Truth')
        plt.imshow(img_2)
        plt.show()
        plt.close()

        return img, img_2

# Defining each convolutional block for the UNet.

# The functions down_block, up_block, bottleneck, and UNet were sourced 
# from this GitHub repository (https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb), 
# with code written by nikhilroxtomar.

# Downsampling stage
def down_block(x, filters, kernelSize=(3,3), padding='same', strides=1):

    c = Conv2D(filters, kernel_size=kernelSize, padding=padding, strides=strides, activation='relu')(x)
    c = Conv2D(filters, kernel_size=kernelSize, padding=padding, strides=strides, activation='relu')(c)
    p = MaxPooling2D((2,2), (2,2))(c)

    return c, p

# Upsampling stage
def up_block(x, skip, filters, kernelSize=(3,3), padding='same', strides=1):

    us = UpSampling2D((2,2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernel_size=kernelSize, padding=padding, strides=strides, activation='relu')(concat)
    c = Conv2D(filters, kernel_size=kernelSize, padding=padding, strides=strides,activation='relu')(c)
    
    return c

# The connecting section between the encoding/downsampling and decoding/upsampling
# sections of the network
def bottleneck(x, filters, kernelSize=(3,3), padding='same', strides=1):

    c = Conv2D(filters, kernel_size=kernelSize, padding=padding, strides=strides, activation='relu')(x)
    c = Conv2D(filters, kernel_size=kernelSize, padding=padding, strides=strides, activation='relu')(c)

    return c

# Defining the architecture of the network in terms of the blocks defined above.

def u_net(inputSize=256):

    f = [16,32,64,128,256]
    inputs = Input((inputSize, inputSize, 1))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])
    c2, p2 = down_block(p1, f[1])
    c3, p3 = down_block(p2, f[2])
    c4, p4 = down_block(p3, f[3])

    bottle_neck = bottleneck(p4, f[4])

    u1 = up_block(bottle_neck, c4, f[3])
    u2 = up_block(u1, c3, f[2])
    u3 = up_block(u2, c2, f[1])
    u4 = up_block(u3, c1, f[0])

    outputs = Conv2D(1, (1,1), padding='same', activation='sigmoid')(u4)
    model = Model(inputs, outputs)

    return model

# A function which creates and saves a loss graph of the validation loss and the 
# training loss over the course of training.

def make_loss_graph(epoch, train_loss, val_loss, save_path):

    graph_save_path = 'path/to/save/directory' + save_path + '/loss_graph'
    x_axis = np.array([])

    for i in range(epoch):
        x_axis = np.append(x_axis, i)

    x_axis = np.array(x_axis)
    val_loss = np.array(val_loss)
    train_loss = np.array(train_loss)

    plt.plot(x_axis, val_loss, label='Validation Loss')
    plt.plot(x_axis, train_loss, label='Training Loss')
    plt.savefig((graph_save_path), dpi=300)
    plt.close()

# The function which trains the segmentation model

def train_model():

    # Definition of the parameters of the training session.
    image_size = 256
    epochs = 200
    batch_size = 16
    data_size = 32
    
    # Define the data generator used to generate the training data
    data = DataGen(batch_size=batch_size, image_size=image_size)
    data.load_training_data()
    
    # Define the model
    model = u_net(inputSize=image_size)
    
    # Uncomment the line below to load an existing, pretrained model
    # model = keras.models.load_model('code/model6/UNet_Model_15.h5')
    
    model.summary()
    
    # Compile the model, using Adam optimiser with the default learning rate and
    # a binary crossentropy loss function.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    start = datetime.now()
    
    # Define variables to store analysis data.
    batchPerEpoch = int(data_size/batch_size)
    loss_log = []
    logs = []
    train_loss = np.array([])
    val_loss = np.array([])

    for i in range(epochs):

        for j in range(batchPerEpoch):

            # Load a batch of training data, train on it, and report the loss and accurary.
            x, y = data.load_training_data()

            loss = model.train_on_batch(
                x, y
            )

            loss_log.append(loss[0])
            print('Epoch: %.d, Batch: %d / %d, Loss: %.8f, ' % ((i+1), (j+1), batchPerEpoch, loss[0]))

       # Evaluate the model at the end of the epoch.
        x, y = data.load_testing_data()
        evaluation_metrics = model.evaluate(x, y, batch_size=1, steps=50)

        mean_loss = sum(loss_log) / len(loss_log)
        train_loss = np.append(train_loss, mean_loss)
        val_loss = np.append(val_loss, evaluation_metrics[0])

        print('Mean loss over the epoch: %.5f' % (mean_loss))

        make_loss_graph(epoch=(i+1), train_loss=train_loss, val_loss=val_loss, save_path='finalModel')

        # Save the model every 15 epochs.
        if (i+1) % 15 == 0:
            print('Saving model...')
            model.save('path/to/save/directory/UNet_Model_%s.h5' % (str(i+1)))
            print('Model saved.')
        
        print(datetime.now())

    print('Number of epochs of training completed:')
    print(epochs)
    print('Start time:')
    print(start)
    print('End time:')
    print(datetime.now())

train_model()