import numpy as np
from os.path import exists, join
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
import pickle
import funkcije
from funkcije import resasample_to_size
from slice_choice_control import get_slices

def standardize(img_np):
    img_np = np.asarray(img_np, dtype=np.float)
    img_np -= np.min(img_np).astype(np.float)
    img_np /= (np.max(img_np).astype(np.float)+0.00001)

    return img_np

def prepare_npy_file(df_of_imgs, slice_plane, choose_slices, image_size):
    # Prepare variables
    iter = 0
    num_of_slices = len(choose_slices)
    ref_df_columns = ['path', 'plane', 'true_label']
    ref_df = pd.DataFrame(columns=ref_df_columns)
    reference_data_idx_name = []
    num_of_3D_imgs = df_of_imgs['pravi3D'].sum()
    slices = np.empty((num_of_3D_imgs * num_of_slices, image_size, image_size))

    # Prepare numpy with slices
    for idx, row in df_of_imgs.iterrows():
        # Load sitk image
        img_sitk = sitk.ReadImage(os.path.join(img_base_path, row['Path']))
        # Write reference data tud dataframe

        ref_df = ref_df.append(pd.DataFrame(data=[[row['Path'], slice_plane, row['sequence']]]*num_of_slices, columns=ref_df_columns, index=[idx]*num_of_slices))
        # Get slices
        slices[iter:iter+3, :, :] = get_slices(funkcije.resample_image_ams(img_sitk, spacing_mm=(1, 1, 1)),
                                            axis=slice_plane,
                                            location=choose_slices)

        iter += num_of_slices
        print('\tProgress: ' + str(np.round(iter/slices.shape[0]*100, 1)) + ' %')
    return slices, ref_df

def save_dataset(npy_X, df_y, description, final_folder_path):
    os.chdir(final_folder_path)
    # npy_X, df_y = mix_order(npy_X, df_y)
    np.save(description + '.npy', npy_X)
    # with open(description + '.txt', 'wb') as fp:
    #     pickle.dump(npy_y, fp)
    df_y.to_pickle(description + '_df')

def mix_order(npy_X, df_y):
    ord = np.arange(npy_X.shape[0])
    np.random.shuffle(ord)
    npy_X = npy_X[ord, :, :]
    df_y = df_y.iloc[ord]
    return npy_X, df_y

## INPUT PARAMETERS
image_size = 128
slice_plane = 'cor'
choose_slices = [-15, 0, 15]
final_folder_name = 'PREPARED_IMAGES_slice_choice_COR_resolution_fixed_levo_desno_fixed'

# File locations
base_path = r'/home/jovyan/shared/InteliRad-gasper'
img_base_path = join(base_path, 'images')
final_folder_path = os.path.join(base_path, final_folder_name)
if not os.path.exists(final_folder_path):
    os.makedirs(final_folder_path)
os.chdir(final_folder_path)

# Import dataframe
features_and_references_file_name = 'features_and_references_dataframe_1866'
features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))

# Get all DeviceSerialNumbers
device_serial_numbers = [['11018'],['45321'], ['70982'], ['21911'], ['000000SI4024MR02'], ['22002'], ['41597'], ['141797'], ['35198'], ['22772', '000000SI4025MR01', '000000SI4024MR01','32283', '67063', '49134', '17260', '35033', '49143', '70826', '35028', '000000007579533T' ]]

## Prepare DATASET C, which consists of 5 imgs from each of dataset in device_serial_numbers list
c_name = 'test_C'
dataset_c_df = pd.DataFrame(columns=features_and_references_dataframe.columns)
for i in device_serial_numbers[:-1]:
    dataset_c_df_tmp = features_and_references_dataframe[features_and_references_dataframe.DeviceSerialNumber.isin(i) & (features_and_references_dataframe.pravi3D == 1)].sample(5)
    dataset_c_df = dataset_c_df.append(dataset_c_df_tmp)
dataset_c_X, dataset_c_y_df = prepare_npy_file(dataset_c_df, slice_plane, choose_slices, image_size)
save_dataset(dataset_c_X, dataset_c_y_df, c_name, final_folder_path)

# Delete imgs in dataset c, so that they can not be repeated in dataset A
features_and_references_dataframe = features_and_references_dataframe.drop(dataset_c_df.index.to_list())

# DATASET A and B
for num, serial_numer in enumerate(device_serial_numbers):
    print('###############     ' + str(num+1) + '/' + str(len(device_serial_numbers)) + '     ###############')
    train_tmp = features_and_references_dataframe[~features_and_references_dataframe.DeviceSerialNumber.isin(serial_numer) & (features_and_references_dataframe.pravi3D == 1)]
    test_tmp = features_and_references_dataframe[features_and_references_dataframe.DeviceSerialNumber.isin(serial_numer) & (features_and_references_dataframe.pravi3D == 1)]

    train_X, train_y_df = prepare_npy_file(train_tmp,
                                               slice_plane,
                                               choose_slices,
                                               image_size)
    test_X, test_y_df = prepare_npy_file(test_tmp,
                                             slice_plane,
                                             choose_slices,
                                             image_size)

    if len(serial_numer) > 1:
        train_file_name = 'train_B'
        test_file_name = 'test_B'
    else:
        train_file_name = 'train' + '_' + str(serial_numer[0])
        test_file_name = 'test' + '_' + str(serial_numer[0])

    save_dataset(train_X, train_y_df, train_file_name, final_folder_path)
    save_dataset(test_X, test_y_df, test_file_name, final_folder_path)
