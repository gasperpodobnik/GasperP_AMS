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

def normalize(img):
    img -= np.min(img)
    img /= np.max(img)
    return img

base_path = r'/home/jovyan/shared/InteliRad-gasper'
imgs_path = r'/home/jovyan/shared/InteliRad-gasper/images'
imgs_folder = os.path.join(base_path, 'few_imgs')
final_folder = os.path.join(base_path, 'Otsu_filter')
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
    img_3d = sitk.ReadImage(img_path)
    nifti = nib.load(img_path)
    img_ni = nil.image.load_img(img_path)
    img_np = sitk.GetArrayFromImage(img_3d)
    img_np = np.moveaxis(img_np, [0,1,2], [2,1,0])
    # img_np = np.fliplr(img_np)
    img_np = normalize(img_np)

    val = filters.threshold_otsu(img_np)
    hist, bins_center = exposure.histogram(img_np)

    img_otsu = (normalize(img_np) > val).astype(np.int)

    display_otsu = plotting.plot_anat(nib.Nifti1Image(img_otsu, nifti.affine), cmap='gray')
    display_normal = plotting.plot_anat(img_ni, cmap='gray', cut_coords=display_otsu.cut_coords)
    display_otsu.savefig(otsu_img_pth)
    display_normal.savefig(normal_img_pth)
    display_otsu.close()
    display_normal.close()

    proj1 = np.fliplr(np.rot90((np.mean(img_otsu, axis=0) > 0).astype(np.int8)))
    proj0 = np.fliplr(np.rot90((np.mean(img_otsu, axis=1) > 0).astype(np.int8)))
    proj2 = np.rot90((np.mean(img_otsu, axis=2) > 0).astype(np.int8), 3)

    proj1 = np.pad(proj1,
                   pad_width=(int(np.floor((np.max(img_otsu.shape) - proj1.shape[0]) / 2)),
                      int(np.ceil((np.max(img_otsu.shape) - proj1.shape[0]) / 2))),
                   mode='constant',
                   constant_values=(0, 0))
    proj0 = np.pad(proj0,
                   pad_width=(int(np.floor((np.max(img_otsu.shape) - proj0.shape[0]) / 2)),
                              int(np.ceil((np.max(img_otsu.shape) - proj0.shape[0]) / 2))),
                   mode='constant',
                   constant_values=(0, 0))
    proj2 = np.pad(proj2,
                   pad_width=(int(np.floor((np.max(img_otsu.shape) - proj2.shape[0]) / 2)),
                              int(np.ceil((np.max(img_otsu.shape) - proj2.shape[0]) / 2))),
                   mode='constant',
                   constant_values=(0, 0))

    projections = np.concatenate((proj0, proj1, proj2), axis=1)
    if projections.shape[1] > 660:
        projections = transform.resize(projections, (projections.shape[0], 660), anti_aliasing=True)
    else:
        projections = np.pad(projections, pad_width=(int(np.floor((660 - projections.shape[1]) / 2)), int(np.ceil((660 - projections.shape[1]) / 2))), mode='constant',
                         constant_values=(0, 0))
    height, width = projections.shape
    dpi = 80
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(projections, interpolation='nearest', cmap='gray')
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    fig.savefig(proj_img_pth, dpi=dpi, transparent=True)

    to_save_np = np.concatenate((np.asarray(Image.open(otsu_img_pth)), np.asarray(Image.open(normal_img_pth)), np.asarray(Image.open(proj_img_pth))), axis=0)
    plt.figure()
    plt.imshow(to_save_np, cmap='gray')
    plt.title(idx)
    plt.savefig(final_img_pth)

    os.remove(otsu_img_pth)
    os.remove(normal_img_pth)
    os.remove(proj_img_pth)

