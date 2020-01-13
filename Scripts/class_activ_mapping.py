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
import keras
from keras.models import Model
import scipy

device_serial_numbers = ['11018','45321', '70982', '21911', '000000SI4024MR02', '22002', '41597', '141797', '35198', 'B', 'C']

model_path = r'eksperimenti_resolution_fixed_levo_desno_fixed//RES_30_12_2019_EPOCHS_20_LR_0.0001_SLICE_MODE_3_CONTRAST=0//11018'
base_path = r'/home/jovyan/shared/InteliRad-gasper'
imgs_folder_name = r'PREPARED_IMAGES_slice_choice_COR_resolution_fixed'
imgs_folder_path = os.path.join(base_path, imgs_folder_name)
final_folder_name = 'CAM_resolution_fixed'
final_folder_path = os.path.join(imgs_folder_path, final_folder_name)
funkcije.create_folder(final_folder_path)
features_and_references_file_name = 'features_and_references_dataframe_1866'
features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))
num_of_slices = 3
rows = 5
cols = 6
modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER'] #['T1W', 'T2W', 'FLAIR', 'OTHER', 'T1W_CONTRAST']

os.chdir(os.path.join(base_path, model_path))
loaded_model = keras.models.load_model('11018_VGG16.h5')

for ser_num in device_serial_numbers:
    os.chdir(imgs_folder_path)
    tmp_df = pd.read_pickle('test_' + ser_num + '_df')
    X_npy = np.load('test_' + ser_num + '.npy')
    izbrani = sample(set(tmp_df.index.to_list()), 30)
    idxs = np.arange(tmp_df.shape[0])[tmp_df.index.isin(izbrani)]
    nepodvojeni = [not(i) for i in tmp_df.loc[izbrani].index.duplicated()]
    idxs = idxs[nepodvojeni]
    true_labels = features_and_references_dataframe.loc[izbrani]['sequence_contrast']

    gap_weights = loaded_model.layers[-1].get_weights()[0]
    cam_model = Model(inputs=loaded_model.input, outputs=(loaded_model.layers[-3].output, loaded_model.layers[-1].output))

    fig, ax = plt.subplots(rows, cols, figsize=(cols*4, rows*4), constrained_layout=True)
    for i, (img_idx, seq, izb_name) in enumerate(zip(idxs, true_labels, izbrani)):
        ax_tmp = ax[i//cols, i%cols]
        ax_tmp.imshow(X_npy[img_idx,:,:], cmap='gray', aspect='equal', alpha=0.5)
        X = np.stack((X_npy[img_idx,:,:],) * 3, axis=-1)
        X = X[np.newaxis,:]

        features, results = cam_model.predict(X)
        pred = np.argmax(results)
        cam_weights = gap_weights[:, pred]
        cam_output = np.dot(features, cam_weights)[0]
        pred_modal = modalitete[pred]

        ax_tmp.imshow(scipy.ndimage.zoom(cam_output, int(128/cam_output.shape[0]), order=3), cmap='jet', alpha=0.5)

        ax_tmp.set_title('True: ' + seq)
        # ax[i // cols, i % cols].axis('off')
        ax_tmp.set_xlabel('Pred: ' + pred_modal)
        ax_tmp.set_ylabel(izb_name)

        edge_color = 'red'
        if seq.lower() == pred_modal.lower() or pred_modal.lower() in seq.lower():
            edge_color = 'green'

        for spine in ax_tmp.spines.values():
            spine.set_linewidth(3)
            spine.set_edgecolor(edge_color)


    plt.savefig(os.path.join(final_folder_path, ser_num))
    plt.close('all')


