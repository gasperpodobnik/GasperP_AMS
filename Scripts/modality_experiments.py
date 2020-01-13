from __future__ import print_function
import os
import glob
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pydot
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import funkcije
import keras
from keras import models
from keras import optimizers
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout, Input
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import tensorflow as tf
import keras_radam


# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

class modality_experiment:
    def __init__(self, contrast_case):
        ## Vhodni parametri:
        # Oganizacijski:
        self.save_results = True
        self.excel_name = 'Results'
        self.excel_folder_name = 'Results'
        self.model_name = 'VGG16'
        self.image_size = 128
        self.experiment_description = 'COR slices eksperimenti'
        # Parametri
        self.NUM_EPOCHS_vgg16 = 25
        self.learning_rate = 1e-4
        self.BATCH_SIZE = 10
        self.optimizer = keras.optimizers.adam(lr=self.learning_rate) #keras_radam.RAdam() # keras.optimizers.adam(lr=self.learning_rate)
        self.validation_set_size = 0.2 # if 0.2, 0.8 of train data will be actually used for training and 0.2 for validation (test dataset is not val dataset)
        self.num_of_slices_per_mri = 3 # known from prepared npy (future: implement automatic calculation of this param.)
        self.npy_mode = 1 # parameter for funkcije.prepare_3_channel_np_arrays
        self.SEED_vgg16 = 49
        if contrast_case == 0:
            self.modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER']
        elif contrast_case == 1:
            self.modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER', 'T1W_CONTRAST']

    def prepare_bureaucracy(self):
        # Po vsej verjetnosti fiksni parametri:
        self.experiment_settings_string = 'VGG16, {} epochs, {}, LR={}, seed={}, npy_mode={}'.format(self.NUM_EPOCHS_vgg16,
                                                                                 self.optimizer, self.learning_rate,
                                                                                 self.SEED_vgg16, self.npy_mode)
        self.npy_folder_name = 'PREPARED_IMAGES_slice_choice_COR_resolution_fixed_levo_desno_fixed'
        self.excel_parameters = ['Input', 'Train_acc', 'Test_acc', 'Train_loss', 'Test_loss', 'ROC_micro']
        self.results_dataframe = pd.DataFrame(columns=self.excel_parameters)
        self.features_and_references_file_name = 'features_and_references_dataframe_1866'
        self.datasets = ['11018','45321', '70982', '21911', '000000SI4024MR02', '22002', '41597', '141797', '35198']
        self.train_file_prefix = 'train' + '_'
        self.test_file_prefix = 'test' + '_'
        self.base_path = r'/home/jovyan/shared/InteliRad-gasper'
        self.img_base_path = os.path.join(self.base_path, 'images')
        self.results_folder_path = os.path.join(self.base_path, self.excel_folder_name)
        self.npy_folder_path = os.path.join(self.base_path, self.npy_folder_name)
        self.model_files_path = os.path.join(self.results_folder_path, self.model_name)
        self.prepare_folders()
        print('\n\n\nNew experiment: ' + self.experiment_description + '\n' + self.experiment_settings_string + '\n\n\n')
        
    def dataset_A_experiment(self):
        self.dataset_letter = 'A'
        for self.num, s in enumerate(self.datasets):
            print('\n' + str(self.num+1) + '/' + str(len(self.datasets)))
            self.dataset = s
            self.dataset_A_and_B_preprocessing()
            self.dataset_files_path = os.path.join(self.results_folder_path, self.dataset)
            funkcije.create_folder(self.dataset_files_path)
            self.train_model()
            self.test_model()

    def dataset_A_and_B_preprocessing(self):
        print('\t' + self.dataset_letter + self.dataset + '\n')
        self.train_npy, self.train_df = funkcije.load_datasets(self.npy_folder_path,
                                                               self.train_file_prefix + self.dataset,
                                                               self.modalitete)
        self.test_npy, self.test_df = funkcije.load_datasets(self.npy_folder_path,
                                                             self.test_file_prefix + self.dataset,
                                                             self.modalitete)
        # Splitting TRAIN and VALIDATION dataset
        self.train_idxs, self.val_idxs = funkcije.train_test_split_patient(self.train_df,
                                                                           val_size=self.validation_set_size,
                                                                           rnd_state=self.SEED_vgg16)
        # Train dataset preprocessing
        self.X_train, self.y_train_df, self.y_train_dummies = self.each_set_preprocess(self.train_idxs,
                                                                                      self.train_df,
                                                                                      self.train_npy)
        # Validation dataset preprocessing
        self.X_val, self.y_val_df, self.y_val_dummies = self.each_set_preprocess(self.val_idxs,
                                                                                      self.train_df,
                                                                                      self.train_npy)
        # Test dataset preprocessing
        self.X_test, self.y_test_df, self.y_test_dummies = self.each_set_preprocess(list(set(self.test_df.index.to_list())),
                                                                                  self.test_df,
                                                                                  self.test_npy)

    def train_model(self):
        self.model = funkcije.initialize_VGG16(mode = self.npy_mode,
                                               num_of_outputs = self.y_train_dummies.shape[1],
                                               image_size = self.image_size)
        self.model, self.history = funkcije.train_VGG16(self.model,
                                          self.X_train,
                                          self.y_train_dummies,
                                          self.X_val,
                                          self.y_val_dummies,
                                          self.optimizer,
                                          self.NUM_EPOCHS_vgg16,
                                          self.BATCH_SIZE)
        # Evaluate model
        self.y_train_predicted = self.model.predict(self.X_train)
        self.elephants_can_remember(self.y_train_dummies, self.y_train_predicted, self.y_train_df, t='train')
        self.score_train = self.model.evaluate(self.X_train, self.y_train_dummies, verbose=0)
        funkcije.save_model(self.model,
                            str(self.dataset) + '_' + self.model_name,
                            self.dataset_files_path)
        funkcije.plot_confusion_matrix(self.y_train_dummies,
                                       self.y_train_predicted,
                                       self.modalitete,
                                       self.dataset_files_path,
                                       title=str(self.dataset) + '_train_cf')
        funkcije.plot_loss_function(self.history,
                                    location=self.dataset_files_path,
                                    title=str(self.dataset) + '_train_graphs')

    def test_model(self):
        self.y_test_predicted = self.model.predict(self.X_test)
        self.elephants_can_remember(self.y_test_dummies, self.y_test_predicted, self.y_test_df, t='test')
        # Evaluate model
        self.score_test = self.model.evaluate(self.X_test, self.y_test_dummies, verbose=0)

        self.roc_micro, _ = funkcije.saveROC(self.y_test_dummies,
                                                self.y_test_predicted,
                                                self.save_results,
                                                self.dataset_files_path,
                                                self.dataset)
        self.results = {'Input': self.dataset,
                   'Input_opis': self.experiment_description,
                   'Modalitete': ', '.join(self.modalitete),
                   'Nastavitve': self.experiment_settings_string,
                   'Train_acc': self.score_train[1],
                   'Test_acc': self.score_test[1],
                   'Train_loss': self.score_train[0],
                   'Test_loss': self.score_test[0],
                   'ROC_micro': self.roc_micro}
        funkcije.plot_confusion_matrix(self.y_test_dummies,
                                       self.y_test_predicted,
                                       self.modalitete,
                                       self.dataset_files_path,
                                       title=str(self.dataset) + '_test_cf')
        self.add_to_results_df()

    def dataset_B_experiment(self):
        self.dataset_letter = 'B'
        self.dataset = 'B'
        self.dataset_A_and_B_preprocessing()
        self.dataset_files_path = os.path.join(self.results_folder_path, self.dataset)
        funkcije.create_folder(self.dataset_files_path)
        self.train_model()
        self.test_model()

    def dataset_C_experiment(self):
        self.dataset_letter = 'C'
        self.dataset = 'C'
        self.dataset_files_path = os.path.join(self.results_folder_path, self.dataset)
        funkcije.create_folder(self.dataset_files_path)
        funkcije.save_model(self.model,
                            str(self.dataset) + '_' + self.model_name,
                            self.dataset_files_path)
        print(self.dataset)
        self.test_npy, self.test_df = funkcije.load_datasets(self.npy_folder_path,
                                                             self.test_file_prefix + self.dataset,
                                                             self.modalitete)
        # Test dataset preprocessing
        self.X_test, self.y_test_df, self.y_test_dummies = self.each_set_preprocess(
            list(set(self.test_df.index.to_list())),
            self.test_df,
            self.test_npy)
        self.test_model()

    def end_experiment(self):
        funkcije.writeResToExcel_from_df(self.excel_name,
                                         [self.experiment_description, self.experiment_settings_string, ', '.join(self.modalitete)],
                                         self.results_dataframe,
                                         self.excel_parameters,
                                         self.results_folder_path,
                                         self.model_files_path)
        plt.close('all')

    def api(self):
        self.prepare_bureaucracy()
        self.dataset_A_experiment()
        self.dataset_B_experiment()
        # eksperimentu B nujno sledi eksperiment C (zaradi uporabe istega natreniranega modela kot pri B)
        self.dataset_C_experiment()
        # Zapiši podatke v excel
        self.end_experiment()

    # Pomožne funkcije
    def prepare_folders(self):
        funkcije.create_folder(self.results_folder_path)
        funkcije.create_folder(self.model_files_path)

    def each_set_preprocess(self, list_of_names, df, npy):
        X, y_df = funkcije.prepare_3_channel_np_arrays(list_of_names,
                                                       df,
                                                       npy,
                                                       num_of_sices=self.num_of_slices_per_mri,
                                                       mode=self.npy_mode)
        if self.npy_mode == 2:  # three same images in three channels
            X = [X[i] for i in range(self.num_of_slices_per_mri)]

        y_dummies = funkcije.to_dummies(y_df, self.modalitete)
        return X, y_df, y_dummies

    def add_to_results_df(self):
        values = [self.results.get(i) for i in self.excel_parameters]
        self.results_dataframe = self.results_dataframe.append(
            pd.DataFrame(data=[values], columns=self.excel_parameters, index=[self.dataset]))

    def elephants_can_remember(self, true_dummies, pred_dummies, df, t = 'test'):
        '''
        This functions stores names of wrongly classified images in dataframe in model_files_path
        :param true_dummies: [ndarray], one-hot-encoded true labels
        :param pred_dummies: [ndarray], one-hot-encoded predicted labels
        :param df: [pandas.Dataframe], Dataframe of reference data (true labels) that has names go imgs as indices
        :return: nothing, just saves Dataframe (to_pickle)
        '''
        modals = np.asarray(self.modalitete)
        true_idx = np.argmax(true_dummies, axis=1)
        pred_idx = np.argmax(pred_dummies, axis=1)
        idxs = true_idx != pred_idx
        archive_df = df.loc[idxs, :]
        if archive_df.shape[0] != 0:
            archive_df = archive_df.assign(pred_label=pd.Series(modals[pred_idx[idxs]]).values)
            archive_df.to_pickle(os.path.join(self.dataset_files_path, self.dataset_letter + '_' + self.dataset + '_' + t.upper() + 'napacno_klasificirani_df'))



