import numpy as np
from os.path import exists, join
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
import pickle
from funkcije import resasample_to_size
from slice_choice_control import get_slices

## INPUT PARAMETERS
image_size = 128
slice_plane = 'cor'
choose_slices = [-15, 0, 15]
final_folder_name = 'PREPARED_IMAGES_slice_choice_COR_novi'

# File locations
base_path = r'/home/jovyan/shared/InteliRad-gasper'
img_base_path = join(base_path, 'images')
final_folder_path = os.path.join(base_path, final_folder_name)
if not os.path.exists(final_folder_path):
    os.makedirs(final_folder_path)
os.chdir(final_folder_path)

# Import dataframe
features_and_references_file_name = 'features_and_references_dataframe_1866'
features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))

# Get all DeviceSerialNumbers
device_serial_numbers = [['11018'],['45321'], ['70982'], ['21911'], ['000000SI4024MR02'], ['22002'], ['41597'], ['141797'], ['35198'], ['22772', '000000SI4025MR01', '000000SI4024MR01','32283', '67063', '49134', '17260', '35033', '49143', '70826', '35028', '000000007579533T' ]]
os.chdir(final_folder_path)
c_df = pd.read_pickle('test_C_df')
c_idx = list(set(c_df.index.to_list()))

def save_dataset(npy_X, df_y, description, final_folder_path):
    os.chdir(final_folder_path)
    npy_X, df_y = mix_order(npy_X, df_y)
    np.save(description + '.npy', npy_X)
    # with open(description + '.txt', 'wb') as fp:
    #     pickle.dump(npy_y, fp)
    df_y.to_pickle(description + '_df')

def mix_order(npy_X, df_y):
    ord = np.arange(npy_X.shape[0])
    np.random.shuffle(ord)
    npy_X = npy_X[ord, :, :]
    df_y = df_y.iloc[ord]
    return npy_X, df_y

def poisci_in_pobrisi(c_idx, X, y_df):
    os.chdir('/home/jovyan/shared/InteliRad-gasper/PREPARED_IMAGES_slice_choice_COR_novi')
    y_out = y_df.copy()
    arhiv = []
    for idx in c_idx:
        indices = [i for i, x in enumerate(y_df.index.to_list()) if x == idx]
        X[indices, :, :] = np.ones((len(indices), 128,128))*(-1)
        y_out.iloc[indices,:] = -1
        arhiv.extend(indices)
    return np.delete(X, arhiv, axis=0), y_out.drop(y_out.index[arhiv])

for num, serial_numer in enumerate(device_serial_numbers):
    if len(serial_numer) == 1:
        X_train = np.load(os.path.join(final_folder_path, 'train_' + serial_numer[0] + '.npy'))
        X_test = np.load(os.path.join(final_folder_path, 'test_' + serial_numer[0] + '.npy'))
        y_train_df = pd.read_pickle(os.path.join(final_folder_path, 'train_' + serial_numer[0] + '_df'))
        y_test_df = pd.read_pickle(os.path.join(final_folder_path, 'test_' + serial_numer[0] + '_df'))
        X_train, y_train_df = poisci_in_pobrisi(c_idx, X_train, y_train_df)
        if X_train.shape[0] == y_train_df.shape[0]:
            save_dataset(X_train, y_train_df, 'train_' + serial_numer[0], final_folder_path)
        X_test, y_test_df = poisci_in_pobrisi(c_idx, X_test, y_test_df)
        if X_test.shape[0] == y_test_df.shape[0]:
            save_dataset(X_test, y_test_df, 'test_' + serial_numer[0], final_folder_path)
    else:
        serial_numer[0] = 'B'
        X_train = np.load(os.path.join(final_folder_path, 'train_' + serial_numer[0] + '.npy'))
        X_test = np.load(os.path.join(final_folder_path, 'test_' + serial_numer[0] + '.npy'))
        y_train_df = pd.read_pickle(os.path.join(final_folder_path, 'train_' + serial_numer[0] + '_df'))
        y_test_df = pd.read_pickle(os.path.join(final_folder_path, 'test_' + serial_numer[0] + '_df'))
        X_train, y_train_df = poisci_in_pobrisi(c_idx, X_train, y_train_df)
        if X_train.shape[0] == y_train_df.shape[0]:
            save_dataset(X_train, y_train_df, 'train_' + serial_numer[0], final_folder_path)
        X_test, y_test_df = poisci_in_pobrisi(c_idx, X_test, y_test_df)
        if X_test.shape[0] == y_test_df.shape[0]:
            save_dataset(X_test, y_test_df, 'test_' + serial_numer[0], final_folder_path)

    
