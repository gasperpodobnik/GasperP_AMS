import numpy as np
from os.path import exists, join
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
import pickle
from funkcije import resasample_to_size
from keras.applications.vgg16 import VGG16
# import matplotlib.pyplot as plt
import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import Callback
from os.path import exists, join
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
from PIL import Image
import collections
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import funkcije


image_size = 128
num_of_slices = 4  # set number of slices per plane (sagittal, axial, coronal) per each image; mora biti sodo
num_of_dim = 3

base_path = r'/home/jovyan/shared/InteliRad-gasper'
img_base_path = os.path.join(base_path, 'images')
npy_base_path = os.path.join(base_path, 'PREPARED_IMAGES')
results_folder = '/home/jovyan/shared/InteliRad-gasper/REZULTATI/'
os.chdir(npy_base_path)

features_and_references_file_name = 'features_and_references_dataframe_1866'
features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))

# datasets = [['11018'],['45321'], ['70982'], ['21911'], ['000000SI4024MR02'], ['22002'], ['41597'], ['141797'], ['35198']]
# datasets = [['DATASET_C']]
datasets = [['45321'],  ['11018'],  ['22002'],  ['141797'],  ['32283'],  ['49143'],  ['67063'],  ['41597'],  ['35198'],  ['22772'],  ['17260'],  ['21911'],  ['000000007579533T'], ['49134'], ['000000SI4024MR02'], ['70982'], ['35028'], ['35033']]

vse_modalitete = ['T1W', 'T2W',	'FLAIR', 'OTHER', 'SPINE_OTHER', 'SPINE_T2W', 'SPINE_T1W', 'SPINE_FLAIR']

for experiment_num, dataset in enumerate(datasets):
    os.chdir(base_path)
    dataset = dataset[0]
    print('##########     ' + str(experiment_num) + '     ##########')
    # train_file_name = 'A_train' + '_' + dataset
    # test_file_name = 'A_test' + '_' + dataset
    test_file_name = dataset
    # if not(os.path.isfile(os.path.join(npy_base_path, train_file_name + '.npy'))):
    #     continue

    _, _, y_test_df = funkcije.load_datasets(npy_base_path, test_file_name)

    y_test_df = y_test_df.loc[~y_test_df.index.duplicated()]

    file_name = 'Dataset_num_of_imgs_C' + '.xlsx'

    if not (os.path.isfile(file_name)):
        writer = pd.ExcelWriter(file_name)  # , engine='xlsxwriter')
        header = ['Serial number', 'Number of imgs']
        header.extend(vse_modalitete)
        pd.DataFrame([header]).to_excel(writer, sheet_name='Sheet1', header=False, index=False)
        writer.save()

    content = pd.read_excel(file_name)
    col_names = content.columns.values
    cont_np = content.values
    cont_np = np.concatenate(([col_names], cont_np))

    number_of_imgs = y_test_df.shape[0]
    stevec_modalitet = np.zeros((len(vse_modalitete)))
    for enum, modaliteta in enumerate(vse_modalitete):
        stevec_modalitet[enum] = int(np.array((y_test_df.sequence == modaliteta).to_list()).sum())
    new_np = [dataset, str(number_of_imgs)]
    new_np.extend([str(i) for i in stevec_modalitet])
    np_to_write = np.concatenate((cont_np, np.asarray(new_np).reshape((1,len(new_np)))))
    to_write = pd.DataFrame(np_to_write)
    writer = pd.ExcelWriter(file_name)
    to_write.to_excel(writer, index=False, header=None)
    writer.save()



