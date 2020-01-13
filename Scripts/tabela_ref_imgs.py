import numpy as np
from os.path import exists, join
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
import pickle
from funkcije import resasample_to_size
from matplotlib import pyplot as plt
from skimage import filters, exposure
from mpl_toolkits.mplot3d import Axes3D
import nilearn as nil
import nibabel as nib
from nilearn import plotting
from PIL import Image
from skimage import transform


'''
Skripta za izračun koliko slik je v kateri kategoriji modalitete po posameznih serijskih ševilkah
'''

mode = 'B'
if mode == 'A':
    device_serial_numbers = ['11018','45321', '70982', '21911', '000000SI4024MR02', '22002', '41597', '141797', '35198', 'B', 'C']
elif mode == 'B':
    device_serial_numbers = ['B']
    subset = [ '22772', '000000SI4025MR01', '000000SI4024MR01','32283', '67063', '49134', '17260', '35033', '49143', '70826', '35028', '000000007579533T' ]
base_path = r'/home/jovyan/shared/InteliRad-gasper'
imgs_folder_name = r'PREPARED_IMAGES_slice_choice_COR_novi'
features_and_references_file_name = 'features_and_references_dataframe_1866'
features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))
modalitete = ['T1W', 'T1W_CONTRAST', 'T2W', 'FLAIR', 'OTHER']
if mode == 'A':
    popis_df = pd.DataFrame(columns=modalitete, index=device_serial_numbers)
elif mode == 'B':
    popis_df = pd.DataFrame(columns=modalitete, index=subset)
num_of_slices = 3

for ser_num in device_serial_numbers:
    os.chdir(os.path.join(base_path, imgs_folder_name))
    tmp_df = pd.read_pickle('test_' + ser_num + '_df')
    if mode == 'A':
        true_labels = features_and_references_dataframe.loc[tmp_df.index, 'sequence_contrast']
        for m in modalitete:
            popis_df.loc[ser_num, m] = sum([1 if i == m else 0 for i in true_labels]) / num_of_slices
    elif mode == 'B':
        for i in subset:
            df_tmp = features_and_references_dataframe.loc[tmp_df.index]
            true_labels = df_tmp.loc[df_tmp['DeviceSerialNumber'] == i]['sequence_contrast']
            for m in modalitete:
                popis_df.loc[i, m] = sum([1 if i == m else 0 for i in true_labels])/num_of_slices


