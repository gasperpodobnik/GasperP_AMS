from __future__ import print_function
from keras.applications.vgg16 import VGG16
# import matplotlib.pyplot as plt
import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.callbacks import Callback
from os.path import exists, join
import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
from PIL import Image
import collections
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn import metrics
import itertools
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import tarfile
import scipy
import random
import shutil
from keras.utils import plot_model
from contextlib import redirect_stdout


vgg3d_model = Sequential()
vgg3d_model.add(Conv3D(input_shape=(100,100,100, 1),filters=64,kernel_size=(3,3,3),padding="same", activation="relu",data_format="channels_last"))
vgg3d_model.add(Conv3D(filters=64, kernel_size=(3,3,3), padding="same", activation="relu"))
vgg3d_model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2)))
vgg3d_model.add(Conv3D(filters=128, kernel_size=(3,3,3), padding="same", activation="relu"))
vgg3d_model.add(Conv3D(filters=128, kernel_size=(3,3,3), padding="same", activation="relu"))
vgg3d_model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2)))
vgg3d_model.add(Conv3D(filters=128, kernel_size=(3,3,3), padding="same", activation="relu"))
vgg3d_model.add(Conv3D(filters=128, kernel_size=(3,3,3), padding="same", activation="relu"))
vgg3d_model.add(Conv3D(filters=128, kernel_size=(3,3,3), padding="same", activation="relu"))
vgg3d_model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2)))
vgg3d_model.add(Conv3D(filters=256, kernel_size=(3,3,3), padding="same", activation="relu"))
vgg3d_model.add(Conv3D(filters=256, kernel_size=(3,3,3), padding="same", activation="relu"))
vgg3d_model.add(Conv3D(filters=256, kernel_size=(3,3,3), padding="same", activation="relu"))
vgg3d_model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2)))
vgg3d_model.add(Conv3D(filters=256, kernel_size=(3,3,3), padding="same", activation="relu"))
vgg3d_model.add(Conv3D(filters=256, kernel_size=(3,3,3), padding="same", activation="relu"))
vgg3d_model.add(Conv3D(filters=256, kernel_size=(3,3,3), padding="same", activation="relu"))
# vgg3d_model.add(MaxPooling3D(pool_size=(2,2,2),strides=(2,2,2)))
vgg3d_model.add(layers.GlobalAveragePooling3D())
vgg3d_model.add(layers.Dense(3, activation='softmax'))
vgg3d_model.summary()


def prepare_X_y(X_in, y_in_df, indices, num_of_imgs=3, image_size = 100, ref_col_name = 'disease_status'):
    X_out = np.zeros((len(indices), image_size, image_size, image_size, 1))
    y_out_df = pd.DataFrame()
    for enum, idx in enumerate(indices):
        slices = np.arange(y_in_df.shape[0])[y_in_df['Image Data ID'] == idx]
        X_out[enum, :,:,:,0] = X_in[slices]
        y_out_df = y_out_df.append(y_in_df.iloc[slices[0]])
    return X_out, y_out_df

adni_folder_path = r'/home/jovyan/shared/mri-brain-prediction/adni'
os.chdir(adni_folder_path)

base_path = '/home/jovyan/shared/InteliRad-gasper'
prepared_folder = 'ADNI_prepared'
npy_path = os.path.join(base_path, prepared_folder)

ref_col_name = 'disease_status'

npy_name = 'adni3d_12_1_2020'

X_all = np.load(os.path.join(npy_path, npy_name + '.npy'))
y_all_df = pd.read_pickle(os.path.join(npy_path, npy_name + '_df'))

num_of_rows = y_all_df.shape[0]
learning_rate = 10e-4

uniq_indices = list(set(y_all_df['Image Data ID']))
idx_train_val, idx_test = train_test_split(uniq_indices, test_size=0.3, random_state=42)
idx_train, idx_val = train_test_split(idx_train_val, test_size=0.15, random_state=42)

X_train, y_train = prepare_X_y(X_all, y_all_df, idx_train)
X_val, y_val = prepare_X_y(X_all, y_all_df, idx_val)
X_test, y_test = prepare_X_y(X_all, y_all_df, idx_test)
y_train_dummies = pd.get_dummies(pd.Categorical(y_train[ref_col_name], categories=[0, 1, 2], ordered=True)).values
y_val_dummies = pd.get_dummies(pd.Categorical(y_val[ref_col_name], categories=[0, 1, 2], ordered=True)).values
y_test_dummies = pd.get_dummies(pd.Categorical(y_test[ref_col_name], categories=[0, 1, 2], ordered=True)).values

del X_all

vgg3d_model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.adam(lr=learning_rate),
              metrics=['acc'])
history = vgg3d_model.fit(X_train, y_train_dummies,
                    epochs=200,
                    validation_data=(X_val, y_val_dummies),
                    batch_size=5)