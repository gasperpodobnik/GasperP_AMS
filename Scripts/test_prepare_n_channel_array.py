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

base_path = '/home/jovyan/shared/InteliRad-gasper'
final_folder_path = os.path.join(base_path, 'TMP_pngs_novi')
if not os.path.exists(final_folder_path):
    os.makedirs(final_folder_path)

dataset_folder_name = 'PREPARED_IMAGES_slice_choice_COR_novi'
dataset_path = os.path.join(base_path, dataset_folder_name)
npy_tmp = np.load(os.path.join(dataset_path, 'test_22002.npy'))
num_of_imgs = npy_tmp.shape[0]
df_tmp = pd.read_pickle(os.path.join(dataset_path, 'test_22002_df'))

plt.figure()
os.chdir(final_folder_path)

train_names, val_names = funkcije.train_test_split_patient(df_tmp, final_folder_path)

out_npy, df_out = funkcije.prepare_3_channel_np_arrays(train_names, df_tmp, npy_tmp, 3, mode=2)