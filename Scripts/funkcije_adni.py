import numpy as np
from os.path import exists, join
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
import pickle
from funkcije import resasample_to_size, make_tarfile
from slice_choice_control import get_slices
from matplotlib import pyplot as plt
import random
import funkcije
import keras
from keras.models import Model
from sklearn.model_selection import train_test_split
import scipy
import imp
imp.reload(funkcije)

def prepare_X_y(X_in, y_in_df, indices, mode, num_of_imgs=3, image_size = 128, ref_col_name = 'disease_status'):
    y_out_df = pd.DataFrame()
    if mode == 1 or mode == 3:
        X_out = np.zeros((len(indices), image_size, image_size, num_of_imgs))
        for enum, idx in enumerate(indices):
            slices = np.arange(y_in_df.shape[0])[y_in_df['Image Data ID'] == idx]
            # print(slices)
            X_out[enum] = np.moveaxis(funkcije.standardize(X_in[slices]), 0, -1)
            y_out_df = y_out_df.append(y_in_df.iloc[slices[0]])
    elif mode == 2:
        X_out_i = np.zeros((3, len(indices), image_size, image_size, num_of_imgs))
        list_tmp = y_in_df.index.to_list()
        for iter, idx in enumerate(indices):
            slices = np.arange(y_in_df.shape[0])[y_in_df['Image Data ID'] == idx]
            y_out_df = y_out_df.append(y_in_df.iloc[slices[0]])
            for pos, s in enumerate(slices):
                X_out_i[pos, iter, :, :, :] = np.stack((funkcije.standardize(X_in[s]),) * 3, axis=-1)
        X_out = [X_out_i[0], X_out_i[1], X_out_i[2]]
    return X_out, y_out_df
