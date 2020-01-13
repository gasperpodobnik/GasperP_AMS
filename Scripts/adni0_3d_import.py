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

def save_dataset(npy_X, df_y, description, final_folder_path):
    os.chdir(final_folder_path)
    # npy_X, df_y = mix_order(npy_X, df_y)
    np.save(description + '.npy', npy_X)
    # with open(description + '.txt', 'wb') as fp:
    #     pickle.dump(npy_y, fp)
    df_y.to_pickle(description + '_df')
    print('SAVED!')

adni_folder_path = r'/home/jovyan/shared/mri-brain-prediction/adni'
os.chdir(adni_folder_path)

adni_std_df = pd.read_excel('data.xlsx', sheet_name='std')
adni_raw_df = pd.read_excel('data.xlsx', sheet_name='raw')

cols = adni_raw_df.columns
adni_raw_df[cols[1]].isin([100741])
only_first_visit_df = adni_raw_df[adni_raw_df['Visit'] == 1]
only_first_visit_df.loc[:, 'disease_status'] = only_first_visit_df.Group
only_first_visit_df.loc[only_first_visit_df.Group == 'CN', 'disease_status'] = 0
only_first_visit_df.loc[only_first_visit_df.Group == 'MCI', 'disease_status'] = 1
only_first_visit_df.loc[only_first_visit_df.Group == 'AD', 'disease_status'] = 2
only_first_visit_df.loc[:, 'rezina'] = only_first_visit_df['Image Data ID']*0

folder_names = only_first_visit_df['Image Data ID'].to_list()
# t1w_img_name = 't1w_RawPreproc_v1.nii.gz'
t1w_img_name = 't1w_SegmentationPreproc_v2.nii.gz'

# slices = [87, 97, 102]
slices = [145, 150, 154] # za coronalne
row_mid, col_mid = (np.array((229, 193))/2).astype(int)
num_of_folders = only_first_visit_df.shape[0]

input_shape = [100, 100, 100]
X_npy = np.zeros((num_of_folders, input_shape[0], input_shape[1], input_shape[2]))
y_df = pd.DataFrame()

npy_idx = 0
for enum, (idx, row) in enumerate(only_first_visit_df.iterrows()):
    folder = row['Image Data ID']
    cur_folder_path = os.path.join(adni_folder_path, str(folder))
    preprocessed_path = os.path.join(cur_folder_path, 'preprocessed')
    if os.path.exists(preprocessed_path) and os.path.exists(os.path.join(preprocessed_path, t1w_img_name)):
        os.chdir(preprocessed_path)
        img_np = sitk.GetArrayFromImage(sitk.ReadImage(t1w_img_name))
        dims = np.asarray(np.array(img_np.shape) / 2, dtype=int)
        X_npy[npy_idx] = img_np[dims[0] - int(input_shape[0]/2):dims[0] + int(input_shape[0]/2),
                 dims[1] - int(input_shape[1]/2):dims[1] + int(input_shape[1]/2),
                 dims[2] - int(input_shape[2]/2):dims[2] + int(input_shape[2]/2)]
        y_df = y_df.append(row)
        npy_idx += 1
        print(str(folder) + '\t' + str(np.round(100 * (enum+1) / (num_of_folders), 2)) + ' %')
    else:
        print('UNSUCCESSFUL: ' + str(folder))

base_path = '/home/jovyan/shared/InteliRad-gasper'
prepared_folder = 'ADNI_prepared'

save_path = os.path.join(base_path, prepared_folder)
funkcije.create_folder(save_path)
save_dataset(X_npy[:y_df.shape[0]], y_df, 'adni3d_12_1_2020', save_path)

# X_out = np.load(os.path.join(save_path, 'adni_12_1_2020' + '.npy'))
# y_out_df = pd.read_pickle(os.path.join(save_path, 'adni_12_1_2020' + '_df'))


