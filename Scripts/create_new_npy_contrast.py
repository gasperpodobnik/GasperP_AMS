import numpy as np
from os.path import exists, join
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
import pickle
from funkcije import resasample_to_size

def standardize(img_np):
    img_np = np.asarray(img_np, dtype=np.float)
    img_np -= np.min(img_np).astype(np.float)
    img_np /= (np.max(img_np).astype(np.float)+0.00001)

    return img_np

def prepare_npy_file(dataframe, image_size, grand_num):
    iter = 0
    reference_data = []
    reference_data_idx_name = []
    num_of_3D_imgs = dataframe['pravi3D'].sum()
    sag_slices = np.empty((num_of_3D_imgs * num_of_slices, image_size, image_size))
    cor_slices = np.empty((num_of_3D_imgs * num_of_slices, image_size, image_size))
    ax_slices = np.empty((num_of_3D_imgs * num_of_slices, image_size, image_size))
    for idx, row in dataframe.iterrows():
        curr_img_fold = row['Path']
        img_true_name = idx.replace(row['Folder'] + '_','')
        img_name_rai = 'RAI_' + img_true_name + '.nii.gz'
        os.chdir(os.path.join(img_base_path, row['Folder']))

        # Check if image is 3D
        img_true = sitk.ReadImage(img_name_rai)
        # img_spacing = img_true.GetSpacing()
        img_size = img_true.GetSize()
        # data_3d = [[img_spacing[0],
        #          img_spacing[1],
        #          img_spacing[2],
        #          img_size[0],
        #          img_size[1],
        #          img_size[2]]]
        # is3D = model_2D_3D.predict(data_3d)[0]
        #
        # if int(is3D) and all(np.array(img_size) > 5):


        reference_data.extend([row['sequence']]*(num_of_slices))
        reference_data_idx_name.extend([idx] * (num_of_slices))
        slices = np.zeros((3, num_of_slices))
        for i, i_dim in enumerate(img_size):
            slices[i, :] = np.round(np.array([np.random.uniform(i_dim*0.2, i_dim*0.42, int(num_of_slices/2)), np.random.uniform(i_dim*0.58, i_dim*0.8, int(num_of_slices/2))]).flatten())
        slices = np.asarray(slices, dtype=np.int)
        for idx_num, idx_slice in enumerate(slices[0,:]): # Sagittal slice
            sag_slices[iter * num_of_slices + idx_num, :, :] = standardize(sitk.GetArrayFromImage(resasample_to_size(img_true[int(idx_slice),:,:])))
        for idx_num, idx_slice in enumerate(slices[1,:]): # Coronal slice
            cor_slices[iter * num_of_slices + idx_num, :, :] = standardize(sitk.GetArrayFromImage(resasample_to_size(img_true[:,int(idx_slice),:])))
        for idx_num, idx_slice in enumerate(slices[2,:]): # Axial slice
            ax_slices[iter * num_of_slices + idx_num, :, :] = standardize(sitk.GetArrayFromImage(resasample_to_size(img_true[:,:,int(idx_slice)])))
        iter += 1
        print(str(grand_num) + ': ' + str(iter))
    img_3d_array = np.concatenate([sag_slices, cor_slices, ax_slices], axis=0)
    ref_out = pd.DataFrame({'sequence': reference_data*3}, index=reference_data_idx_name*3)
    return img_3d_array, reference_data*3, ref_out

def save_dataset(npy_X, npy_y, df_y, description, serial_number):
    os.chdir(os.path.join(base_path, 'PREPARED_IMAGES_CONTRAST'))
    np.save(description + '_' + serial_number + '.npy', npy_X)
    with open(description + '_' + serial_number + '.txt', 'wb') as fp:
        pickle.dump(npy_y, fp)
    df_y.to_pickle(description + '_' + serial_number + '_df')

image_size = 128
num_of_slices = 4  # set number of slices per plane (sagittal, axial, coronal) per each image; mora biti sodo

base_path = r'/home/jovyan/shared/InteliRad-gasper'
img_base_path = join(base_path, 'images')

# Load the model for 2D/3D segmentation
model_2D_3D_filename = '2D_3D_model.sav'
model_2D_3D = pickle.load(open(os.path.join(base_path,model_2D_3D_filename), 'rb'))

# Get list of all image folders
fold_list = np.array([])
for i in os.walk(img_base_path):
    fold_list = np.append(fold_list, i[0])
fold_list = fold_list[1:]

# Import dataframes
features_and_references_file_name = 'features_and_references_dataframe_1866'
features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))

# Get all DeviceSerialNumbers
# device_serial_numbers = set(features_and_references_dataframe[features_and_references_dataframe.pravi3D == 1]['DeviceSerialNumber'].to_list())
device_serial_numbers = [['11018'], ['22002'], ['141797'], ['21911'], ['70982'], ['32283', '49143', '67063', '41597', '35198', '22772', '17260'], ['45321', '000000007579533T', '49134', '000000SI4024MR02', '35028', '35033']]
dataset_names = ['A', 'A', 'A', 'A', 'A', 'B1', 'B2']

# DATASET A and B
for num, serial_numer in enumerate(device_serial_numbers):
    print('###############     ' + str(num) + '     ###############')
    train_tmp = features_and_references_dataframe[~features_and_references_dataframe.DeviceSerialNumber.isin(serial_numer) & (features_and_references_dataframe.pravi3D == 1) & (features_and_references_dataframe.sequence == 'T1W')]
    test_tmp = features_and_references_dataframe[features_and_references_dataframe.DeviceSerialNumber.isin(serial_numer) & (features_and_references_dataframe.pravi3D == 1) & (features_and_references_dataframe.sequence == 'T1W')]

    train_X_tmp, train_y_tmp, train_y_tmp_df = prepare_npy_file(train_tmp, image_size, num)
    test_X_tmp, test_y_tmp, test_y_tmp_df = prepare_npy_file(test_tmp, image_size, num)

    p1 = np.arange(train_X_tmp.shape[0])
    np.random.shuffle(p1)
    train_X = train_X_tmp[p1, :, :]
    train_y = [train_y_tmp[i] for i in p1]
    p2 = np.arange(test_X_tmp.shape[0])
    np.random.shuffle(p2)
    test_X = test_X_tmp[p2, :, :]
    test_y = [test_y_tmp[i] for i in p2]

    if len(serial_numer) > 1:
        file_name = 'zdruzeno'
    else:
        file_name = str(serial_numer[0])

    save_dataset(train_X, train_y, train_y_tmp_df, 'train', dataset_names[num] + '_' + file_name)
    save_dataset(test_X, test_y, test_y_tmp_df, 'test', dataset_names[num] + '_' + file_name)
