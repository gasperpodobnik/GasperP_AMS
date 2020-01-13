import numpy as np
import os

from funkcije import saveImage

base_path = r'/home/jovyan/shared/InteliRad-gasper'
os.chdir(base_path)

X = np.load('img_3d_array.npy')

saveImage(X[3,:,:], 'rezina4')