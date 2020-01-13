import numpy as np
from os.path import exists, join
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
import pickle
import funkcije
from funkcije import resasample_to_size
from matplotlib import pyplot as plt
from skimage import filters, exposure
from mpl_toolkits.mplot3d import Axes3D
import nilearn as nil
import nibabel as nib
from nilearn import plotting
from PIL import Image
from skimage import transform

def normalize(img):
    img -= np.min(img)
    img /= np.max(img).astype(np.float)
    return img

def extract_slice(array3d, axis, slice):
    '''
    :param array3d: 3d matrix from which slice will be taken along given axis
    :param axis: Integer or string ('sag', 'cor', 'ax'), axis along which slice will taken in array3d
    :param slice: Index of slice which will be taken from array3d
    :return: Slice along given axis in array3d
    '''
    if axis == 0 or axis == 'sag':
        o_array = array3d[slice,:,:]
    elif axis == 1 or axis == 'cor':
        o_array = array3d[:, slice, :]
    elif axis == 2 or axis == 'ax':
        o_array = array3d[:, :, slice]
    return o_array

def calc_mean_slice(proj):
    sp_meja = np.argmax(proj)
    zg_meja = len(proj) - 1 - np.argmax(proj[::-1])
    sred = np.mean([sp_meja, zg_meja]).astype(int)
    return [sp_meja, zg_meja], sred

def get_slices(img_sitk, axis, location, pad_shift=128):
    '''
    Get normalized slices along given axis
    :param img_sitk: 3d SimpleITK IMAGE
    :param axis: String ('sag', 'cor', 'ax'), axis along which the slices will be taken
    :param location: List of floats, percents around the middle slice, -15 means, mean_slice -15/100*(slice_max - slice_min) (slice_(min, max and mean) are calculated using Otsu filter
    :return: 2d numpy arrays, slices from 3d img
    '''
    img = sitk.GetArrayFromImage(img_sitk).astype(np.float)
    img = np.moveaxis(img, [0, 1, 2], [2, 1, 0])  # (sag, cor, ax)
    img = normalize(img)
    val = filters.threshold_otsu(img)
    hist, _ = exposure.histogram(img)
    img_otsu = (img > val).astype(np.int)

    if axis == 'sag':
        proj = (np.max(img_otsu, axis=2)).astype(np.int8)
        [slice_min, slice_max], slice_mean = calc_mean_slice(np.max(proj, axis=1))
    elif axis == 'cor':
        proj = (np.max(img_otsu, axis=2)).astype(np.int8)
        [slice_min, slice_max], slice_mean = calc_mean_slice(np.max(proj, axis=0))
        [slice_min_sag, slice_max_sag], slice_mean_sag = calc_mean_slice(np.max(proj, axis=1))
        proj_ax = np.rot90((np.max(img_otsu, axis=1)).astype(np.int8), 3)
        [slice_min_ax, slice_max_ax], slice_mean_ax = calc_mean_slice(np.max(proj_ax, axis=1))
    elif axis == 'ax':
        proj = np.rot90((np.max(img_otsu, axis=1)).astype(np.int8), 3)
        [slice_min, slice_max], slice_mean = calc_mean_slice(np.max(proj, axis=1))
    else:
        raise ValueError("Axis argument must be one of ('sag', 'cor', 'ax')")
    slices = np.zeros((len(location), 128, 128))
    for enum, s in enumerate(location): # dela samo za axis = 'cor'
        img_tmp = extract_slice(img, axis, int(slice_mean + s/100*(slice_max-slice_min)))
        img_tmp = np.pad(img_tmp, pad_shift, funkcije.pad_with, padder=0)
        gor = pad_shift + slice_mean_sag - 64
        dol = pad_shift + slice_mean_sag + 64
        levo = pad_shift + slice_max_ax - 128
        desno = pad_shift + slice_max_ax
        if (slice_max_ax - 128) < 0:
            levo = pad_shift
            desno = levo + 128
        slices[enum, :, :] = normalize(img_tmp[gor:dol,
                                       levo:desno])
    return slices

def main():
    base_path = r'/home/jovyan/shared/InteliRad-gasper'
    imgs_path = r'/home/jovyan/shared/InteliRad-gasper/images'
    imgs_folder = os.path.join(base_path, 'few_imgs')
    final_folder = os.path.join(base_path, 'Pregled_rezin')
    otsu_img_pth = os.path.join(final_folder, 'otsu.png')
    normal_img_pth = os.path.join(final_folder, 'normal.png')
    proj_img_pth = os.path.join(final_folder, 'proj.png')
    if not os.path.exists(final_folder): os.mkdir(final_folder)

    features_and_references_file_name = 'features_and_references_dataframe_1866'
    features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))
    df_tmp = features_and_references_dataframe[features_and_references_dataframe.pravi3D == 1].sample(100)

    for idx, row in df_tmp.iterrows():
        print(idx)
        final_img_pth = os.path.join(final_folder, idx + '.png')
        cur_folder = os.path.join(imgs_path, df_tmp.loc[idx, 'Folder'])
        img_path = os.path.join(cur_folder, 'RAI_' + df_tmp.loc[idx, 'Path'].split('\\')[1])
        img_sitk = sitk.ReadImage(img_path)
        img_np = sitk.GetArrayFromImage(img_sitk)
        img_np = np.moveaxis(img_np, [0,1,2], [2,1,0]) # (sag, cor, ax)
        img_np = normalize(img_np)

        val = filters.threshold_otsu(img_np)
        hist, bins_center = exposure.histogram(img_np)

        img_otsu = (img_np > val).astype(np.int)

        proj_sag = np.rot90((np.max(img_otsu, axis=0)).astype(np.int8), 3)
        proj_cor = np.rot90((np.max(img_otsu, axis=1)).astype(np.int8), 3)
        proj_ax = (np.max(img_otsu, axis=2)).astype(np.int8)

        [cor_sp, cor_zg], cor_mean = calc_mean_slice(np.max(proj_ax, axis=0))
        [sag_sp, sag_zg], sag_mean = calc_mean_slice(np.max(proj_ax, axis=1))
        [ax_sp, ax_zg], ax_mean = calc_mean_slice(np.max(proj_cor, axis=1))

        projections = [proj_sag, proj_cor, proj_ax]
        slice_means = [proj_cor, proj_ax, proj_sag]

        fig = plt.figure()
        # Sag mask
        i = 1
        plt.subplot(2,3,i)
        plt.imshow(proj_sag, origin='lower', aspect='auto')
        plt.axis('off')
        plt.plot([cor_sp] * 2, [0, proj_sag.shape[0]-1], 'g')
        plt.plot([cor_mean] * 2, [0, proj_sag.shape[0] - 1])
        plt.plot([cor_zg] * 2, [0, proj_sag.shape[0] - 1], 'g')
        plt.plot([0, proj_sag.shape[1]-1], [ax_sp]*2, 'g')
        plt.plot([0, proj_sag.shape[1] - 1], [ax_mean]*2)
        plt.plot([0, proj_sag.shape[1]-1], [ax_zg]*2, 'g')
        plt.subplot(2,3,3+i)
        plt.imshow(extract_slice(img_np, 1, cor_mean), aspect='auto')
        plt.axis('off')

        # Cor mask
        i = 2
        plt.subplot(2,3,i)
        plt.imshow(proj_cor, origin='lower', aspect='auto')
        plt.axis('off')
        plt.plot([sag_sp] * 2, [0, proj_cor.shape[0]-1], 'g')
        plt.plot([sag_mean] * 2, [0, proj_cor.shape[0] - 1])
        plt.plot([sag_zg] * 2, [0, proj_cor.shape[0] - 1], 'g')
        plt.plot([0, proj_cor.shape[1]-1], [ax_sp]*2, 'g')
        plt.plot([0, proj_cor.shape[1] - 1], [ax_mean]*2)
        plt.plot([0, proj_cor.shape[1]-1], [ax_zg]*2, 'g')
        plt.subplot(2,3,3+i)
        plt.imshow(extract_slice(img_np, 2, ax_mean), aspect='auto')
        plt.axis('off')

        # Ax mask
        i = 3
        plt.subplot(2,3,i)
        plt.imshow(proj_ax, origin='lower', aspect='auto')
        plt.axis('off')
        plt.plot([cor_sp] * 2, [0, proj_ax.shape[0]-1], 'g')
        plt.plot([cor_mean] * 2, [0, proj_ax.shape[0] - 1])
        plt.plot([cor_zg] * 2, [0, proj_ax.shape[0] - 1], 'g')
        plt.plot([0, proj_sag.shape[1]-1], [sag_sp]*2, 'g')
        plt.plot([0, proj_sag.shape[1] - 1], [sag_mean]*2)
        plt.plot([0, proj_sag.shape[1]-1], [sag_zg]*2, 'g')
        plt.subplot(2,3,3+i)
        plt.imshow(extract_slice(img_np, 0, sag_mean), aspect='auto')
        plt.axis('off')

        fig.suptitle(str(idx))
        plt.savefig(os.path.join(final_folder, idx + '.png'))
        plt.close()

if __name__== "__main__":
    main()