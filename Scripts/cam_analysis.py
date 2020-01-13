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

base_path = r'/home/jovyan/shared/InteliRad-gasper'
adni_folder_path = r'/home/jovyan/shared/mri-brain-prediction/adni'
os.chdir(adni_folder_path)

## INPUT PARAMETERS
cam_folder_name = 'CAM_analysis'
prepared_imgs_folder_name = 'PREPARED_IMAGES_slice_choice_COR_resolution_fixed_levo_desno_fixed'
mode = 3
num_of_slices_per_mri = 3

# File locations
img_base_path = join(base_path, 'images')
cam_folder_path = os.path.join(base_path, cam_folder_name)
prepared_imgs_folder_path = os.path.join(base_path, prepared_imgs_folder_name)
funkcije.create_folder(cam_folder_path)
os.chdir(cam_folder_path)

old_expriment_path0 = os.path.join(base_path, 'eksperimenti_3_1_2020_vsi')
mode1_path = os.path.join(old_expriment_path0, 'RES_3_1_2020_EPOCHS_20_LR_0.0001_SLICE_MODE_1_CONTRAST=0')
mode2_path = os.path.join(old_expriment_path0, 'RES_3_1_2020_EPOCHS_20_LR_0.0001_SLICE_MODE_2_CONTRAST=0')
mode3_path = os.path.join(old_expriment_path0, 'RES_3_1_2020_EPOCHS_20_LR_0.0001_SLICE_MODE_3_CONTRAST=0')
model_path = os.path.join(mode3_path, 'B')

modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER']
# modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER', 'T1W_CONTRAST']

train_dataset_names = ['11018', '45321', '70982', '21911', '000000SI4024MR02', '22002', '41597', '141797', '35198', 'B']
loaded_model = keras.models.load_model(os.path.join(model_path, 'B_VGG16.h5'))
cam_name = 'CAM_imgs'
num_of_datasets = len(train_dataset_names)

for enum, dataset_name in enumerate(train_dataset_names):
    X_raw_npy, y_raw_df = funkcije.load_datasets(prepared_imgs_folder_path, 'test_' + dataset_name, modalitete)
    indices = list(set(y_raw_df.index.to_list()))
    X_npy, y_df, y_dummies = funkcije.each_set_preprocess_for_modalities(mode,
                                                                  indices,
                                                                  y_raw_df,
                                                                  X_raw_npy,
                                                                  num_of_slices_per_mri,
                                                                  modalitete)
    num_of_imgs = X_npy.shape[0]
    X_cam = np.zeros((X_npy.shape[:-1]))
    y_cam = []
    for slice in range(num_of_imgs):
        X_cam[slice], y_tmp = funkcije.create_cam(loaded_model, X_npy[slice], already_3channel=True)
        y_cam.extend(y_tmp)
        print(str(np.round(100 * (slice + 1) / num_of_imgs, 2)) + ' %')
    y_df['pred'] = y_cam
    funkcije.save_dataset(X_cam, y_df,
                          description=cam_name + '_' + dataset_name,
                          final_folder_path=cam_folder_path)
    print('\tFinished: ' + str((enum+1)/num_of_datasets))


rows = (num_of_datasets + 1)*2 # ker posebej izris za prave modalitete in napačne +1 ker skupno še
cols = len(modalitete)
fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
alpha = 0.3

X_concatenated_cam = np.zeros((1, 128, 128))
X_concatenated_npy = np.zeros((1, 128, 128))
y_concatenated = pd.DataFrame()

for enum_dataset, dataset_name in enumerate(train_dataset_names):
    file_name = cam_name + '_' + dataset_name
    X_cam = np.load(os.path.join(cam_folder_path, file_name + '.npy'))
    X_npy = np.load(os.path.join(prepared_imgs_folder_path, 'test_' + dataset_name + '.npy'))
    y_cam_df = pd.read_pickle(os.path.join(cam_folder_path, file_name + '_df'))
    num_of_imgs = X_cam.shape[0]

    for enum_modal, modal in enumerate(modalitete):
        for enum_col, col in enumerate(y_cam_df.columns):
            ax_tmp = ax[2*enum_dataset+enum_col, enum_modal]
            mask = y_cam_df[col] == modal
            y_df_specific = y_cam_df[mask]
            idxs = np.arange(num_of_imgs)[mask]
            if idxs.size == 0:
                img_background = np.zeros_like(X_npy[0])
                X_cam_specific = np.zeros_like(img_background)
            else:
                npy_idx = np.arange(num_of_imgs)[y_cam_df.index == y_df_specific.index[0]][0]
                img_background = X_npy[npy_idx]
                X_cam_specific = np.mean(funkcije.normalize_CAM(X_cam[idxs]), axis=0)
            ax_tmp.imshow(img_background, cmap='gray', aspect='equal', alpha=(1 - alpha))
            ax_tmp.imshow(X_cam_specific, cmap='jet', alpha=alpha)
            ax_tmp.set_title('True: ' + modal)
            ax_tmp.set_xlabel('True/Pred: ' + col)
            ax_tmp.set_ylabel('Dataset: ' + dataset_name)
    X_concatenated_cam = np.concatenate((X_concatenated_cam, X_cam), axis=0)
    X_concatenated_npy = np.concatenate((X_concatenated_npy, X_npy), axis=0)
    y_concatenated = y_concatenated.append(y_cam_df)
# še za vse skupaj
X_cam = X_concatenated_cam[1:]
X_npy = X_concatenated_npy[1:]
y_cam_df = y_concatenated

description = cam_name + '_' + 'vsi'
os.chdir(cam_folder_path)
np.save(description + '.npy', X_cam)
y_cam_df.to_pickle(description + '_df')

os.chdir(prepared_imgs_folder_path)
np.save('test_vsi' + '.npy', X_npy)


dataset_name = 'vsi_skupaj'
num_of_imgs = X_cam.shape[0]
for enum_modal, modal in enumerate(modalitete):
    for enum_col, col in enumerate(y_cam_df.columns):
        ax_tmp = ax[2*num_of_datasets+enum_col, enum_modal]
        mask = y_cam_df[col] == modal
        y_df_specific = y_cam_df[mask]
        npy_idx = np.arange(num_of_imgs)[y_cam_df.index == y_df_specific.index[0]][0]
        idxs = np.arange(num_of_imgs)[mask]
        if idxs.size == 0:
            img_background = np.zeros_like(X_npy[0])
            X_cam_specific = np.zeros_like(img_background)
        else:
            img_background = X_npy[npy_idx]
            X_cam_specific = np.mean(funkcije.normalize_CAM(X_cam[idxs]), axis=0)
        ax_tmp.imshow(img_background, cmap='gray', aspect='equal', alpha=(1 - alpha))
        ax_tmp.imshow(X_cam_specific, cmap='jet', alpha=alpha)
        ax_tmp.set_title('True: ' + modal)
        ax_tmp.set_xlabel('True/Pred: ' + col)
        ax_tmp.set_ylabel('Dataset: ' + dataset_name)

plt.savefig(os.path.join(cam_folder_path, 'CAM_po_datasetih'))
plt.close('all')

plt.figure()
plt.imshow(img_background, cmap='gray', aspect='equal', alpha=(1 - alpha))
plt.imshow(np.mean(funkcije.normalize_CAM(X_cam), axis=0), cmap='jet', alpha=alpha)
plt.savefig(os.path.join(cam_folder_path, 'CAM_povp_vsi'))
plt.close('all')


