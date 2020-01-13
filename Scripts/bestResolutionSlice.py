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
num_of_dims = 1
NP_FILE_NAME = '{}slices_{}dim_Gauss_sag'.format(num_of_slices, num_of_dims)
SAVE = True




base_path = r'/home/jovyan/shared/InteliRad-gasper'

img_fold_path = r'/home/jovyan/shared/InteliRad-gasper/images'
os.chdir(img_fold_path)
fold_list = np.array([])
for i in os.walk(img_fold_path):
    fold_list = np.append(fold_list, i[0])
fold_list = fold_list[1:]

all_imgs = glob.glob("*.nii.gz") # list with image names

feat_name = r'features_df_1866'
feat_data = pd.read_pickle(os.path.join(base_path,feat_name))
ref_name = r'references_df_1866'
ref_data = pd.read_pickle(os.path.join(base_path,ref_name)) # indexes are img names, values are reference sequences

# load the model for 2D/3D segmentation from disk
model_2D_3D_filename = '2D_3D_model.sav'
model_2D_3D = pickle.load(open(os.path.join(base_path,model_2D_3D_filename), 'rb'))
img_3d_array = np.zeros((1044*num_of_dims*num_of_slices, image_size,image_size))
iter = 0
ref_sync = []
headers = list(feat_data.columns.values)
headers.append('ref_sequence')
headers.append('ref_contrast')
izhodni_df = pd.DataFrame([], columns=headers)

for idx, row in feat_data.iterrows():
    curr_img_fold = row['Folder']
    img_true_name = idx.replace(curr_img_fold + '_','')
    img_name_rai = 'RAI_' + img_true_name + '.nii.gz'
    os.chdir(os.path.join(img_fold_path, curr_img_fold))

    img_true = sitk.ReadImage(img_name_rai)
    img_spacing = img_true.GetSpacing()
    img_size = img_true.GetSize()
    data_3d = [[img_spacing[0],
             img_spacing[1],
             img_spacing[2],
             img_size[0],
             img_size[1],
             img_size[2]]]
    result = bool(int(model_2D_3D.predict(data_3d)[0]))

    if result:
        mdl_ctr = ''.join([ref_data.loc[idx, 'sequence'], ref_data.loc[idx, 'hasContrast (0/1)']])
        new_line = list(row)
        new_line.append(ref_data.loc[idx, 'sequence'])
        new_line.append(ref_data.loc[idx, 'hasContrast (0/1)'])
        izhodni_df = izhodni_df.append(pd.DataFrame([new_line]*(num_of_dims*num_of_slices), columns=headers))
        ref_sync.extend([mdl_ctr]*(num_of_dims*num_of_slices))
        slices = np.zeros((3, num_of_slices))
        # for i, i_dim in enumerate(img_size):
        #     slices[i,:] = np.round(np.random.normal(i_dim/2, i_dim/4, num_of_slices))
        #     slices[i, :][slices[i, :] >= i_dim] = i_dim/2
        # slices = np.asarray(slices, dtype=np.int)
        # # for idx_num, idx_slice in enumerate(slices[2,:]):
        # #     img_3d_array[iter*num_of_dims*num_of_slices + idx_num,:,:] = standardize(sitk.GetArrayFromImage(resasample_to_size(img_true[:,:,int(idx_slice)])))
        # for idx_num, idx_slice in enumerate(slices[0,:]):
        #     img_3d_array[iter*num_of_dims*num_of_slices+idx_num,:,:] = standardize(sitk.GetArrayFromImage(resasample_to_size(img_true[int(idx_slice),:,:])))
        # for idx_num, idx_slice in enumerate(slices[1,:]):
        #     img_3d_array[iter*num_of_dims*num_of_slices + idx_num + num_of_slices,:,:] = standardize(sitk.GetArrayFromImage(resasample_to_size(img_true[:,int(idx_slice),:])))
        # for idx_num, idx_slice in enumerate(slices[2,:]):
        #     img_3d_array[iter*num_of_dims*num_of_slices + idx_num + 2*num_of_slices,:,:] = standardize(sitk.GetArrayFromImage(resasample_to_size(img_true[:,:,int(idx_slice)])))
        iter += 1


        print(iter)
print('Final:' + str(iter))


if SAVE:
    os.chdir(os.path.join(base_path, 'PREPARED_IMAGES'))
    # np.save(NP_FILE_NAME + '.npy', img_3d_array)
    #
    # with open(os.path.join(base_path, NP_FILE_NAME + '_ref.txt'), 'wb') as fp:
    #     pickle.dump(ref_sync, fp)

    izhodni_df.to_pickle(NP_FILE_NAME + '_df')

