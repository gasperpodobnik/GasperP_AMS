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



# File locations
base_path = r'/home/jovyan/shared/InteliRad-gasper'
img_base_path = join(base_path, 'images')
folder = 'MR01'
img_name = 'MR eFLAIR 3D SENSE.nii.gz'
img_name2 = 'MR MPR COR.nii.gz'
img_folder_path = os.path.join(img_base_path, folder)
img_path = os.path.join(img_folder_path, img_name)
img_path2 = os.path.join(img_folder_path, img_name2)

img_sitk = sitk.ReadImage(img_path)
img_sitk2 = sitk.ReadImage(img_path2)

img_sitk_out = funkcije.resample_image_ams(img_sitk2)

funkcije.saveImage(sitk.GetArrayFromImage(img_sitk2)[:,30,:], 'originalna')
funkcije.saveImage(sitk.GetArrayFromImage(img_sitk_out)[:,30,:], 'resampled')

