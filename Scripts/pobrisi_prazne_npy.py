import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import keras
import pickle
import funkcije
from funkcije import writeResToExcel, saveROC
from sklearn.utils import shuffle

base_path = r'/home/jovyan/shared/InteliRad-gasper'
img_base_path = os.path.join(base_path, 'images')
npy_base_path = os.path.join(base_path, 'PREPARED_IMAGES')
os.chdir(npy_base_path)



for experiment_num in range(24):
    print('##########     ' + str(experiment_num))
    train_file_name = 'ucni_podatki_1_train' + str(experiment_num)
    test_file_name = 'ucni_podatki_1_test' + str(experiment_num)

    X_train = np.load(os.path.join(npy_base_path, train_file_name + '.npy'))
    with open(os.path.join(npy_base_path, train_file_name + '.txt'), 'rb') as fp:
        y_train = pickle.load(fp)

    test_file_name2 = [filename for filename in os.listdir(npy_base_path) if filename.startswith(test_file_name + '_')][0][:-4]
    opis_slik = str(test_file_name2.split('_')[-1])
    print('     ' + opis_slik + '     ##########')
    X_test = np.load(os.path.join(npy_base_path, test_file_name2 + '.npy'))
    with open(os.path.join(npy_base_path, test_file_name + '.txt'), 'rb') as fp:
        y_test = pickle.load(fp)



    if X_test.shape[0] == 0:
        os.chdir(npy_base_path)
        os.remove(test_file_name2 + '.npy')
        os.remove(test_file_name + '.txt')
        os.remove(train_file_name + '.npy')
        os.remove(train_file_name + '.txt')
        print(test_file_name2)
