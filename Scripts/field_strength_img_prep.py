from __future__ import print_function
import numpy as np
# import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import Callback
from os.path import exists, join
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
from funkcije import train, resasample_to_size
import pickle
from matplotlib import pyplot as plt

def standardize(img_np):
    img_np = np.asarray(img_np, dtype=np.float)
    img_np -= np.min(img_np).astype(np.float)
    img_np /= np.max(img_np).astype(np.float)

    return img_np
print('Notebook run using keras:', keras.__version__)





image_size = 128
num_of_slices = 4
num_of_dims = 3
NP_FILE_NAME = '{}slices_{}dim_Gauss_CC_359'.format(num_of_slices, num_of_dims)
SAVE = True

base_path = r'/home/jovyan/shared/InteliRad-gasper'

img_fold_path = r'/home/jovyan/shared/CC-359/Original'
os.chdir(img_fold_path)

all_imgs = os.listdir()
all_imgs.remove('.DS_Store')

# load the model for 2D/3D segmentation from disk
model_2D_3D_filename = '2D_3D_model.sav'
model_2D_3D = pickle.load(open(os.path.join(base_path,model_2D_3D_filename), 'rb'))
img_3d_array = np.zeros((359*num_of_dims*num_of_slices, image_size,image_size))
iter = 0
ref_sync = []
# headers = list(feat_data.columns.values)
# headers.append('ref_sequence')
# headers.append('ref_contrast')
# izhodni_df = pd.DataFrame([], columns=headers)
features_names = ['Image_num', 'vendor', 'field_strength', 'age', 'sex']
features_df = pd.DataFrame(columns=features_names)


for img_name in all_imgs:
    img_true_name = img_name
    img_name_rai = 'prep_' + img_true_name + '.nii.gz'
    os.chdir(img_fold_path)

    img_true = sitk.ReadImage(img_true_name)
    img_spacing = img_true.GetSpacing()
    img_size = img_true.GetSize()
    data_3d = [[img_spacing[0],
             img_spacing[1],
             img_spacing[2],
             img_size[0],
             img_size[1],
             img_size[2]]]
    result = bool(int(model_2D_3D.predict(data_3d)[0]))

    img_features = img_true_name[:-7].split('_')
    img_number = img_features[0]
    vendor = img_features[1]
    field_strength = img_features[2]
    age = img_features[3]
    sex = img_features[4]

    new_row = [img_number, vendor, field_strength, age, sex]
    old = features_df.values.tolist()
    old.extend([new_row] * (num_of_dims * num_of_slices))
    features_df = pd.DataFrame(old, columns=features_names)

    slices = np.zeros((3, num_of_slices))
    for i, i_dim in enumerate(img_size):
        slices[i,:] = np.round(np.random.normal(i_dim/2, i_dim/4, num_of_slices))
        slices[i, :][slices[i, :] >= i_dim] = i_dim/2
    slices = np.asarray(slices, dtype=np.int)
    # for idx_num, idx_slice in enumerate(slices[2,:]):
    #     img_3d_array[iter*num_of_dims*num_of_slices + idx_num,:,:] = standardize(sitk.GetArrayFromImage(resasample_to_size(img_true[:,:,int(idx_slice)])))
    for idx_num, idx_slice in enumerate(slices[0,:]):
        img_3d_array[iter*num_of_dims*num_of_slices+idx_num,:,:] = standardize(sitk.GetArrayFromImage(resasample_to_size(img_true[int(idx_slice),:,:])))
    for idx_num, idx_slice in enumerate(slices[1,:]):
        img_3d_array[iter*num_of_dims*num_of_slices + idx_num + num_of_slices,:,:] = standardize(sitk.GetArrayFromImage(resasample_to_size(img_true[:,int(idx_slice),:])))
    for idx_num, idx_slice in enumerate(slices[2,:]):
        img_3d_array[iter*num_of_dims*num_of_slices + idx_num + 2*num_of_slices,:,:] = standardize(sitk.GetArrayFromImage(resasample_to_size(img_true[:,:,int(idx_slice)])))
    iter += 1


    print(iter)
print('Final:' + str(iter))


if SAVE:
    os.chdir(os.path.join(base_path, 'PREPARED_IMAGES'))
    np.save(NP_FILE_NAME + '.npy', img_3d_array)

    with open(os.path.join(base_path, NP_FILE_NAME + '_ref.txt'), 'wb') as fp:
        pickle.dump(ref_sync, fp)

    features_df.to_pickle(NP_FILE_NAME + '_df')


