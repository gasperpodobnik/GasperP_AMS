from keras.utils.vis_utils import plot_model
from keras import models
from keras import layers
from keras import optimizers
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import numpy as np
import keras
from keras import Input
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Activation, Dropout, Flatten, Dense, AveragePooling2D


image_size = 128
num_classes = 4

def create_convnet(img_path='network_image.png'):
    input_shape = Input(shape=(image_size, image_size, 1))

    tower_1 = Conv2D(20, (3, 3), activation='relu')(input_shape)
    tower_1 = Conv2D(20, (3, 3), activation='relu')(tower_1)
    tower_1 = Conv2D(20, (3, 3), activation='relu')(tower_1)
    # tower_1 = MaxPooling2D((1, 11), strides=(1, 1), padding='same')(tower_1)

    tower_2 = Conv2D(20, (3, 3), activation='relu')(input_shape)
    tower_2 = MaxPooling2D((5, 5))(tower_2)

    tower_3 = AveragePooling2D((7, 7), activation='relu')(input_shape)
    tower_3 = MaxPooling2D((5, 5))(tower_3)

    merged = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    merged = Flatten()(merged)

    out = AveragePooling2D((2, 2), activation='relu')(merged)
    out = Dense(128, activation='relu')(out)
    out = Dense(num_classes, activation='softmax')(out)

    model = keras.Model(input_shape, out)
    plot_model(model, to_file=img_path)

    return model