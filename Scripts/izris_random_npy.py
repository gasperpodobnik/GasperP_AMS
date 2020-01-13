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
from random import sample
import funkcije

device_serial_numbers = ['11018','45321', '70982', '21911', '000000SI4024MR02', '22002', '41597', '141797', '35198', 'B', 'C']

base_path = r'/home/jovyan/shared/InteliRad-gasper'
imgs_folder_name = r'PREPARED_IMAGES_slice_choice_COR_resolution_fixed_levo_desno_fixed'
imgs_folder_path = os.path.join(base_path, imgs_folder_name)
final_folder_name = 'Pregled_random_30_2'
final_folder_path = os.path.join(imgs_folder_path, final_folder_name)
funkcije.create_folder(final_folder_path)
features_and_references_file_name = 'features_and_references_dataframe_1866'
features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))
num_of_slices = 3
rows = 5
cols = 6

for ser_num in device_serial_numbers:
    os.chdir(imgs_folder_path)
    tmp_df = pd.read_pickle('test_' + ser_num + '_df')
    X_npy = np.load('test_' + ser_num + '.npy')
    izbrani = sample(set(tmp_df.index.to_list()), 30)
    idxs = []
    for i in izbrani:
        idxs.extend(np.arange(tmp_df.shape[0])[tmp_df.index.isin([i])])
    nepodvojeni = [not(i) for i in tmp_df.loc[izbrani].index.duplicated()]
    idxs = np.array(idxs)[nepodvojeni]
    true_labels = features_and_references_dataframe.loc[izbrani]['sequence_contrast']
    fig, ax = plt.subplots(rows, cols, figsize=(cols*4, rows*4), constrained_layout=True)
    for i, (img_idx, seq, izb_name) in enumerate(zip(idxs, true_labels, izbrani)):
        ax_tmp = ax[i//cols, i%cols]
        ax_tmp.imshow(X_npy[img_idx,:,:], cmap='gray', aspect='equal')
        ax_tmp.set_title(seq)
        # ax_tmp.axis('off')
        ax_tmp.set_xlabel(izb_name)
    plt.savefig(os.path.join(final_folder_path, ser_num))
    plt.close('all')


