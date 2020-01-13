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

def prepare_X_y(X_in, y_in_df, indices, mode=1, num_of_imgs=3, image_size = 128, ref_col_name = 'disease_status'):
    y_out_df = pd.DataFrame()
    if mode == 1 or mode == 3:  # three different slices in three channels of image
        X_out = np.zeros((len(indices), image_size, image_size, num_of_imgs))
        for enum, idx in enumerate(indices):
            slices = np.arange(y_in_df.shape[0])[y_in_df['path'] == idx]
            X_out[enum] = np.moveaxis(X_in[slices], 0, -1)
            y_out_df = y_out_df.append(y_in_df.iloc[slices[0]])
    elif mode == 2:  # three same images in three channels
        list_tmp = y_in_df['path'].to_list()
        X_out = np.empty((num_of_imgs, len(indices), image_size, image_size, 3))
        for iter, idx in enumerate(indices):
            y_out_df = y_out_df.append(pd.DataFrame(data=[y_in_df['true_label'].iloc[list_tmp.index(idx)]],
                                                columns=['true_label'],
                                                index=[idx]))
            for s in range(num_of_imgs):
                num = list_tmp.index(idx)
                X_out[s, iter, :, :, :] = np.stack((X_in[num,:,:],) * 3, axis=-1)
                list_tmp[num] = None
    return X_out, y_out_df

def elephants_can_remember(true_dummies, pred_dummies, df, save_path, t='vsi'):
    '''
    This functions stores names of wrongly classified images in dataframe in model_files_path
    :param true_dummies: [ndarray], one-hot-encoded true labels
    :param pred_dummies: [ndarray], one-hot-encoded predicted labels
    :param df: [pandas.Dataframe], Dataframe of reference data (true labels) that has names go imgs as indices
    :return: nothing, just saves Dataframe (to_pickle)
    '''
    modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER']
    modals = np.asarray(modalitete)
    true_idx = np.argmax(true_dummies, axis=1)
    pred_idx = np.argmax(pred_dummies, axis=1)
    idxs = true_idx != pred_idx
    archive_df = df.loc[idxs, :]
    if archive_df.shape[0] != 0:
        archive_df = archive_df.assign(pred_label=pd.Series(modals[pred_idx[idxs]]).values)
        archive_df.to_pickle(os.path.join(save_path + t + 'napacno_klasificirani_df'))

adni_folder_path = r'/home/jovyan/shared/mri-brain-prediction/adni'
os.chdir(adni_folder_path)

## INPUT PARAMETERS
final_folder_name = 'PREPARED_IMAGES_adni_for_modality'

# File locations
base_path = r'/home/jovyan/shared/InteliRad-gasper'
img_base_path = join(base_path, 'images')
final_folder_path = os.path.join(base_path, final_folder_name)
funkcije.create_folder(final_folder_path)
os.chdir(final_folder_path)

dataset_name = 'vsi_adni_7_1_2020'
# dataset_name = 'vse_preiskave_adni_7_1_20207'
X_vsi = np.load(os.path.join(final_folder_path, dataset_name + '.npy'))
y_vsi_df = pd.read_pickle(os.path.join(final_folder_path, dataset_name + '_df'))

indices_vsi = list(set(y_vsi_df['path']))

vsi_X, vsi_y_df = prepare_X_y(X_vsi, y_vsi_df, indices_vsi, mode=3)

old_expriment_path0 = os.path.join(base_path, 'eksperimenti_3_1_2020_vsi')
old_expriment_path1_mode1 = os.path.join(old_expriment_path0, 'RES_3_1_2020_EPOCHS_20_LR_0.0001_SLICE_MODE_1_CONTRAST=0')
old_expriment_path1_mode2 = os.path.join(old_expriment_path0, 'RES_3_1_2020_EPOCHS_20_LR_0.0001_SLICE_MODE_2_CONTRAST=0')
old_expriment_path1_mode3 = os.path.join(old_expriment_path0, 'RES_3_1_2020_EPOCHS_20_LR_0.0001_SLICE_MODE_3_CONTRAST=0')
old_expriment_path2 = os.path.join(old_expriment_path1_mode3, 'B')

loaded_model = keras.models.load_model(os.path.join(old_expriment_path2, 'B_VGG16.h5'))
# X = [vsi_X[0], vsi_X[1], vsi_X[2]]
X = vsi_X
modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER']
# modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER', 'T1W_CONTRAST']
y_true_vsi_dummies = pd.get_dummies(pd.Categorical(vsi_y_df['true_label'], categories=modalitete, ordered=True)).values
y_pred_vsi_dummies = loaded_model.predict(X)



elephants_can_remember(y_true_vsi_dummies, y_pred_vsi_dummies, vsi_y_df, final_folder_path, t='test')
# Evaluate model
score_test = loaded_model.evaluate(X, y_true_vsi_dummies, verbose=0)

roc_micro, _ = funkcije.saveROC(y_true_vsi_dummies,
                                        y_pred_vsi_dummies,
                                        True,
                                        final_folder_path,
                                        dataset_name)

funkcije.plot_confusion_matrix(y_true_vsi_dummies,
                               y_pred_vsi_dummies,
                               modalitete,
                               final_folder_path,
                               title=str(dataset_name) + '_test_cf')

X_cam = np.zeros_like(X[:,:,:,0])
y_cam = []
for slice in range(X.shape[0]):
    X_cam[slice], y_tmp = funkcije.create_cam(loaded_model, X[slice], already_3channel=True)
    y_cam.append(y_tmp)
    print(str(100*(slice+1)/X.shape[0]) + ' %')

X_norm = X_cam/np.sum(np.sum(X_cam, axis=1), axis=1)[:, np.newaxis, np.newaxis]

def saveCAM(img_np, img_mri, name):
    base_path = r'/home/jovyan/shared/InteliRad-gasper'
    fig, ax = plt.subplots(1, 1)
    alpha = 0.4
    ax.imshow(img_mri, cmap='gray', alpha=(1-alpha))
    im = ax.imshow(img_np, cmap='jet', alpha=alpha)
    plt.colorbar(im, ax=ax)
    os.chdir(os.path.join(base_path, 'ogledSlik'))
    plt.savefig(name)
    plt.close('all')
    return print('Image saved!')

saveCAM(np.mean(X_norm, axis=0), X[0, :, :, 0], 'mean_cam_vsi_mri')
os.chdir(final_folder_path)
np.save('normalizirani_CAM_adni_vsi_contrast0' + '.npy', X_norm)
