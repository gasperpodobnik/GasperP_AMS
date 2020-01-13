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
from sklearn.model_selection import train_test_split
import keras

def save_dataset(npy_X, df_y, description, final_folder_path):
    os.chdir(final_folder_path)
    # npy_X, df_y = mix_order(npy_X, df_y)
    np.save(description + '.npy', npy_X)
    # with open(description + '.txt', 'wb') as fp:
    #     pickle.dump(npy_y, fp)
    df_y.to_pickle(description + '_df')
    print('SAVED!')

def prepare_npy_file(df_of_imgs, slice_plane, choose_slices, image_size=128):
    # Prepare variables
    adni_folder_path = r'/home/jovyan/shared/mri-brain-prediction/adni'
    iter = 0
    seq = 'T1W'
    t1w_img_name = 't1w_RawPreproc_v1.nii.gz'
    num_of_slices = len(choose_slices)
    ref_df_columns = ['path', 'plane', 'true_label']
    ref_df = pd.DataFrame(columns=ref_df_columns)
    reference_data_idx_name = []
    num_of_3D_imgs = df_of_imgs['Image Data ID'].shape[0]
    slices = np.empty((num_of_3D_imgs * num_of_slices, image_size, image_size))

    # Prepare numpy with slices
    for idx, row in df_of_imgs.iterrows():
        folder = row['Image Data ID']
        cur_folder_path = os.path.join(adni_folder_path, str(folder))
        preprocessed_path = os.path.join(cur_folder_path, 'preprocessed')
        if os.path.exists(preprocessed_path) and os.path.exists(os.path.join(preprocessed_path, t1w_img_name)):
            os.chdir(preprocessed_path)
            # Load sitk image
            img_sitk = sitk.ReadImage(t1w_img_name)

            # Write reference data tud dataframe
            ref_df = ref_df.append(pd.DataFrame(data=[[folder, slice_plane, seq]]*num_of_slices, columns=ref_df_columns, index=[idx]*num_of_slices))
            # Get slices
            slices[iter:iter+3, :, :] = get_slices(funkcije.resample_image_ams(img_sitk, spacing_mm=(1, 1, 1)),
                                                axis=slice_plane,
                                                location=choose_slices)

            iter += num_of_slices
            print('\tProgress: ' + str(np.round(iter/slices.shape[0]*100, 1)) + ' %')
    return slices, ref_df

adni_folder_path = r'/home/jovyan/shared/mri-brain-prediction/adni'
os.chdir(adni_folder_path)

adni_std_df = pd.read_excel('data.xlsx', sheet_name='std')
adni_raw_df = pd.read_excel('data.xlsx', sheet_name='raw')

cols = adni_raw_df.columns
adni_raw_df[cols[1]].isin([100741])
only_first_visit_df = adni_raw_df[adni_raw_df['Visit'] == 1]
only_first_visit_df['true_label'] = 'T1W'

folder_names = only_first_visit_df['Image Data ID'].to_list()
t1w_img_name = 't1w_RawPreproc_v1.nii.gz'

## INPUT PARAMETERS
image_size = 128
slice_plane = 'cor'
choose_slices = [-15, 0, 15]
final_folder_name = 'PREPARED_IMAGES_adni_for_modality'

# File locations
base_path = r'/home/jovyan/shared/InteliRad-gasper'
img_base_path = join(base_path, 'images')
final_folder_path = os.path.join(base_path, final_folder_name)
funkcije.create_folder(final_folder_path)
os.chdir(final_folder_path)


uniq_indices = list(set(only_first_visit_df['Image Data ID']))
idx_train, idx_test = train_test_split(uniq_indices, test_size=0.3, random_state=42)
train_tmp = only_first_visit_df[only_first_visit_df['Image Data ID'].isin(idx_train)]
test_tmp = only_first_visit_df[only_first_visit_df['Image Data ID'].isin(idx_test)]

train_X, train_y_df = prepare_npy_file(train_tmp,
                                       slice_plane,
                                       choose_slices,
                                       image_size)
test_X, test_y_df = prepare_npy_file(test_tmp,
                                     slice_plane,
                                     choose_slices,
                                     image_size)
vsi_X, vsi_y_df = prepare_npy_file(only_first_visit_df,
                                     slice_plane,
                                     choose_slices,
                                     image_size)

save_dataset(train_X[:train_y_df.shape[0]], train_y_df, 'train_adni_7_1_2020', final_folder_path)
save_dataset(test_X[:test_y_df.shape[0]], test_y_df, 'test_adni_7_1_2020', final_folder_path)
save_dataset(vsi_X[:vsi_y_df.shape[0]], vsi_y_df, 'vse_preiskave_adni_7_1_20207', final_folder_path)
