import numpy as np
from os.path import exists, join
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
import pickle
from funkcije import resasample_to_size

npy_file_name = 'ucni_podatki'

base_path = r'/home/jovyan/shared/InteliRad-gasper'
img_base_path = join(base_path, 'images')

# Import dataframes
features_and_references_file_name = 'features_and_references_dataframe_1866'
features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))

for idx, row in features_and_references_dataframe.iterrows():
    if features_and_references_dataframe.loc[idx, 'maxSpacing'] <= 3.3 and features_and_references_dataframe.loc[idx, 'minSliceNum'] >= 20:
        features_and_references_dataframe.loc[idx, 'pravi3D'] = 1
    else:
        features_and_references_dataframe.loc[idx, 'pravi3D'] = 0

features_and_references_dataframe.to_pickle(features_and_references_file_name)