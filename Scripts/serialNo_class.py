from __future__ import print_function
import os
import glob
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
import itertools
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



class serialNo_experiment:
    def __init__(self):
        self.excel_name = 'Results'
        self.base_path = r'/home/jovyan/shared/InteliRad-gasper'
        self.results_folder = '/home/jovyan/shared/InteliRad-gasper/REZULTATI60/'
        self.img_base_path = os.path.join(self.base_path, 'images')
        self.npy_base_path = os.path.join(self.base_path, 'PREPARED_IMAGES_slice_choice_COR_novi')
        self.save_results = True
        self.features_and_references_file_name = 'features_and_references_dataframe_1866'
        self.datasets = ['11018','45321', '70982', '21911', '000000SI4024MR02', '22002', '41597', '141797', '35198']
        self.train_file_prefix = 'train' + '_'
        self.test_file_prefix = 'test' + '_'
        self.final_folder_suffix = ''
        self.model_name_suffix = ''
        self.num_of_slices_per_mri = 3
        self.npy_mode = 1

    def without_contrast(self):
        self.contrast_case = 0
        self.modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER']

    def only_t1w_with_contrast(self):
        self.contrast_case = 1
        self.modalitete = ['T1W', 'T1W_CONTRAST']
        self.npy_base_path += '_CONTRAST'
        self.final_folder_suffix = '_' + 'T1W_and_T1W_CONTRAST'
        self.model_name_suffix = '_' + 'T1W_and_T1W_CONTRAST'
        self.datasets = ['11018', '22002', '141797', '21911', '70982', 'zdruzeno1', 'zdruzeno2']

    def all_modals_with_contrast(self):
        self.contrast_case = 2
        self.modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER', 'T1W_CONTRAST']
        self.final_folder_suffix = '_' + 'CONTRAST_ALL_MODAL'
        self.model_name_suffix = '_' + 'CONTRAST_ALL_MODAL'

    def final_folder(self, final_folder_name):
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        self.res_path = os.path.join(self.results_folder, final_folder_name + self.final_folder_suffix)
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)

    def set_model_name(self, model_name):
        self.model_name = model_name + self.model_name_suffix

    def plan_experiment(self):
        self.image_size = 128
        self.features_and_references_dataframe = pd.read_pickle(os.path.join(self.base_path, self.features_and_references_file_name))
        self.features_and_references_dataframe['sequence'] = self.features_and_references_dataframe['sequence'].mask(~self.features_and_references_dataframe.sequence.isin(self.modalitete), 'OTHER')
        self.modalitete_encoded = np.arange(len(self.modalitete)).astype(int)
        self.dicom_parameters()

    def dicom_parameters(self):
        self.features = ['RepetitionTime',
                         'EchoTime',
                         'InversionTime',
                         'MagneticFieldStrength',
                         'SliceThickness',
                         'NumberofAverages',
                         'ImagingFrequency',
                         'NumberofPhaseEncodingSteps',
                         'EchoTrainLength',
                         'PercentSampling',
                         'PercentPhaseFieldofView',
                         'PixelBandwidth',
                         'FlipAngle']

    def vgg16_parameters(self):
        self.set_model_name('VGG16')
        self.final_folder('VGG16_experiment')
        self.SEED_vgg16 = 49
        self.NUM_EPOCHS_vgg16 = 2
        self.LR_vgg16 = 1e-4
        self.nastavitve = 'VGG16, {} epochs, RMSprop(lr={}), seed={}, npy_mode={}'.format(self.NUM_EPOCHS_vgg16,
                                                                                 self.LR_vgg16,
                                                                                 self.SEED_vgg16, self.npy_mode)
        self.plan_experiment()

    def mlp_parameters(self):
        self.set_model_name('MLP')
        self.final_folder('MLP_experiment')
        self.num_of_neurons_in_hidden_layer_mlp = 20
        self.NUM_EPOCHS_mlp = 25
        self.dropout_rate_mlp = 0.5
        self.actv_fun_mlp = 'relu'
        self.optimizer_algorithm_mlp = 'adam'
        self.nastavitve = 'MLP, {} epochs, dropout rate: {}, {} neurons in ' \
                              'hidden layer, actv. fun {}, optimizer: {}'.format(self.NUM_EPOCHS_mlp,
                                                                                 self.dropout_rate_mlp,
                                                                                 self.num_of_neurons_in_hidden_layer_mlp,
                                                                                 self.actv_fun_mlp,
                                                                                 self.optimizer_algorithm_mlp)
        self.plan_experiment()

    def vgg16_and_mlp_parameters(self):
        self.set_model_name('VGG16_and_MLP')
        self.final_folder('VGG16_and_MLP_experiment')
        self.SEED_vgg16 = 49
        self.NUM_EPOCHS_vgg16 = 25
        self.LR_vgg16 = 1e-4
        self.nastavitve_vgg16 = 'VGG16, {} epochs, RMSprop(lr={}), seed={}'.format(self.NUM_EPOCHS_vgg16,
                                                                                 self.LR_vgg16,
                                                                                 self.SEED_vgg16)
        self.num_of_neurons_in_hidden_layer_mlp = 20
        self.NUM_EPOCHS_mlp = 25
        self.dropout_rate_mlp = 0.5
        self.actv_fun_mlp = 'relu'
        self.optimizer_algorithm_mlp = 'adam'
        self.nastavitve_mlp = 'MLP, {} epochs, dropout rate: {}, {} neurons in ' \
                              'hidden layer, actv. fun {}, optimizer: {}'.format(self.NUM_EPOCHS_mlp,
                                                                                 self.dropout_rate_mlp,
                                                                                 self.num_of_neurons_in_hidden_layer_mlp,
                                                                                 self.actv_fun_mlp,
                                                                                 self.optimizer_algorithm_mlp)
        self.LR_vgg_and_mlp = 1e-4
        self.nastavitve = self.nastavitve_vgg16 + '\n' + self.nastavitve_mlp
        self.plan_experiment()

    def rf_parameters(self):
        self.set_model_name('RF')
        self.final_folder('RF_experiment')
        self.N_ESTIMATORS = 100
        self.CRITERION = 'entropy'
        self.MIN_S_SPLIT = 60
        self.nastavitve = 'RF, n_estimators: {}, criterion: {}, min sample split: {}'.format(self.N_ESTIMATORS, self.CRITERION, self.MIN_S_SPLIT)
        self.plan_experiment()

class VGG16_and_MLP:
    def __init__(self, serialNo_class):
        self.settings = serialNo_class

    def dataset_A_experiment(self):
        self.dataset_letter = 'A'
        for self.num, self.dataset in enumerate(self.settings.datasets):
            self.train_model()

    def dataset_B_and_C_experiment(self):
        self.dataset_letter = 'B'
        self.train_model()
        self.dataset_letter = 'C'
        self.prepare_for_training()
        self.analyse_model()

    def prepare_for_training(self):
        prepare_dataset(self)
        # spremenljivke brez oznake se nanašajo na vgg16 (in ne na mlp)
        # Import data for mlp model
        self.train_tmp = self.settings.features_and_references_dataframe.loc[self.y_train_df.index.to_list()]
        self.test_tmp = self.settings.features_and_references_dataframe.loc[self.y_test_df.index.to_list()]

        self.X_train_mlp, self.y_train_mlp, self.y_train_mlp_df = funkcije.get_img_params(
            self.train_tmp.loc[list(set(self.y_train_df.index.to_list()))],
            self.settings.features, self.settings.modalitete, self.settings.modalitete_encoded)
        self.X_test_mlp, self.y_test_mlp, self.y_test_mlp_df = funkcije.get_img_params(
            self.test_tmp.loc[list(set(self.y_test_df.index.to_list()))],
            self.settings.features, self.settings.modalitete, self.settings.modalitete_encoded)
        prepare_contrast(self)

    def train_model(self):
        self.prepare_for_training()
        self.X_train, self.X_validation, \
        self.y_train_dummies, self.y_validation_dummies, \
        self.y_train_df, self.y_validation_df, \
        self.X_train_mlp, self.X_validation_mlp = train_test_split(self.X_train,
                                                                   self.y_train_dummies,
                                                                   self.y_train_df,
                                                                   self.X_train_mlp,
                                                                   random_state=self.settings.SEED_vgg16)
        #### MODEL
        # Model architecture
        mlp_inputs = Input(shape=(self.X_train_mlp.shape[1],))
        x = Dense(self.settings.num_of_neurons_in_hidden_layer_mlp, activation=self.settings.actv_fun_mlp)(mlp_inputs)
        x = Dropout(rate=self.settings.dropout_rate_mlp)(x)
        # x = Dense(num_of_neurons_in_hidden_layer, activation='sigmoid', kernel_initializer='random_normal')(x)
        # x = Dropout(rate=dropout_rate)(x)
        mlp_out = Dense(len(self.settings.modalitete), activation='sigmoid', kernel_initializer='random_normal')(x)

        self.vgg_cnn = keras.applications.vgg16.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(self.settings.image_size, self.settings.image_size, 3))

        # Freeze the layers except the last 4 layers
        for layer in self.vgg_cnn.layers[:-4]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in self.vgg_cnn.layers:
            print(layer, layer.trainable)

        self.vgg_cnn.summary()

        cnn_inputs = Input(shape=self.X_train.shape[1:], )

        vgg_out = self.vgg_cnn(cnn_inputs)

        vgg_flattened = Flatten()(vgg_out)
        merged = keras.layers.concatenate(inputs=[vgg_flattened, mlp_out], axis=1)
        combined = Dense(50, activation='sigmoid')(merged)
        combined = Dropout(rate=self.settings.dropout_rate_mlp)(combined)
        final_layer = Dense(len(self.settings.modalitete), activation='relu')(combined)

        self.model = Model(inputs=[cnn_inputs, mlp_inputs], outputs=final_layer)
        self.model.summary()

        os.chdir(self.settings.res_path)
        keras.utils.plot_model(self.model, to_file='model.png')

        self.model.compile(loss='categorical_crossentropy',
                             optimizer=optimizers.RMSprop(lr=self.settings.LR_vgg_and_mlp),
                             metrics=['acc'])

        self.model.fit([self.X_train, self.X_train_mlp],
                         y=self.y_train_dummies,
                         epochs=self.settings.NUM_EPOCHS_mlp,
                         verbose=1,
                         shuffle=True,
                         validation_data=([self.X_validation, self.X_validation_mlp], self.y_validation_dummies))
        self.analyse_model()

    def analyse_model(self):
        self.y_train_predicted = self.model.predict([self.X_train, self.X_train_mlp])
        self.y_test_predicted = self.model.predict([self.X_test, self.X_test_mlp])

        self.score_train = self.model.evaluate([self.X_train, self.X_train_mlp], self.y_train_dummies, verbose=0)
        self.score_test = self.model.evaluate([self.X_test, self.X_test_mlp], self.y_test_dummies, verbose=0)

        print('Train loss: {}, train accuracy: {}'.format(self.score_train[0], self.score_train[1]))
        print('Test loss: {}, test accuracy: {}'.format(self.score_test[0], self.score_test[1]))

        statistical_analysis(self)

        save_results(self)

class RF:
    def __init__(self, serialNo_class):
        self.settings = serialNo_class

    def dataset_A_experiment(self):
        self.dataset_letter = 'A'
        for self.num, self.dataset in enumerate(self.settings.datasets):
            self.train_model()

    def dataset_B_and_C_experiment(self):
        self.dataset_letter = 'B'
        self.train_model()
        self.dataset_letter = 'C'
        self.prepare_for_training()
        self.analyse_model()

    def prepare_for_training(self):
        prepare_dataset(self)
        self.train_tmp = self.settings.features_and_references_dataframe.loc[self.y_train_df.index.to_list()]
        self.test_tmp = self.settings.features_and_references_dataframe.loc[self.y_test_df.index.to_list()]

        self.X_train, self.y_train_df = funkcije.get_img_params(
            self.train_tmp.loc[list(set(self.train_tmp.index.to_list()))],
            self.settings.features, self.settings.modalitete, self.settings.modalitete_encoded)
        self.X_test, self.y_test_df = funkcije.get_img_params(
            self.test_tmp.loc[list(set(self.test_tmp.index.to_list()))],
            self.settings.features, self.settings.modalitete, self.settings.modalitete_encoded)
        prepare_contrast(self)

    def train_model(self):
        self.prepare_for_training()
        self.model = RandomForestClassifier(n_estimators=self.settings.N_ESTIMATORS, criterion=self.settings.CRITERION, min_samples_split=self.settings.MIN_S_SPLIT)
        self.model.fit(self.X_train, self.y_train_dummies)
        self.analyse_model()

    def analyse_model(self):
        self.y_train_predicted = self.model.predict(self.X_train)
        self.y_test_predicted = self.model.predict(self.X_test)

        self.score_train = ['', self.model.score(self.X_train, self.y_train_dummies)]
        self.score_test = ['', self.model.score(self.X_test, self.y_test_dummies)]

        statistical_analysis(self)

        save_results(self)

class MLP:
    def __init__(self, serialNo_class):
        self.settings = serialNo_class

    def dataset_A_experiment(self):
        self.dataset_letter = 'A'
        for self.num, self.dataset in enumerate(self.settings.datasets):
            self.train_model()

    def dataset_B_and_C_experiment(self):
        self.dataset_letter = 'B'
        self.train_model()
        self.dataset_letter = 'C'
        self.prepare_for_training()
        self.analyse_model()

    def prepare_for_training(self):
        prepare_dataset(self)
        self.train_tmp = self.settings.features_and_references_dataframe.loc[self.y_train_df.index.to_list()]
        self.test_tmp = self.settings.features_and_references_dataframe.loc[self.y_test_df.index.to_list()]

        self.X_train, self.y_train_df = funkcije.get_img_params(
            self.train_tmp.loc[list(set(self.train_tmp.index.to_list()))],
            self.settings.features, self.settings.modalitete, self.settings.modalitete_encoded)
        self.X_test, self.y_test_df = funkcije.get_img_params(
            self.test_tmp.loc[list(set(self.test_tmp.index.to_list()))],
            self.settings.features, self.settings.modalitete, self.settings.modalitete_encoded)
        prepare_contrast(self)

    def train_model(self):
        self.prepare_for_training()
        self.model = Sequential()
        # First Hidden Layer
        self.model.add(Dense(self.settings.num_of_neurons_in_hidden_layer_mlp,
                             activation=self.settings.actv_fun_mlp,
                             kernel_initializer='random_normal',
                             input_dim=len(self.settings.features)))
        self.model.add(Dropout(self.settings.dropout_rate_mlp))
        # Output Layer
        self.model.add(Dense(len(self.settings.modalitete),
                             activation='sigmoid',
                             kernel_initializer='random_normal'))
        self.model.add(Dropout(self.settings.dropout_rate_mlp))
        # Compiling the neural network
        self.model.compile(optimizer=self.settings.optimizer_algorithm_mlp,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        # Fitting the data to the training dataset
        self.model.fit(self.X_train, self.y_train_dummies, epochs=self.settings.NUM_EPOCHS_mlp)
        self.analyse_model()

    def analyse_model(self):
        self.y_train_predicted = self.model.predict(self.X_train)
        self.y_test_predicted = self.model.predict(self.X_test)

        self.score_train = self.model.evaluate(self.X_train, self.y_train_dummies, verbose=0)
        self.score_test = self.model.evaluate(self.X_test, self.y_test_dummies, verbose=0)

        statistical_analysis(self)

        save_results(self)

class pretrained_VGG16:
    def __init__(self, serialNo_class):
        self.settings = serialNo_class

    def dataset_A_experiment(self):
        self.dataset_letter = 'A'
        for self.num, self.dataset in enumerate(self.settings.datasets):
            print(self.dataset)
            self.train_model()

    def dataset_B_and_C_experiment(self):
        self.dataset_letter = 'B'
        print(self.dataset)
        self.train_model()
        self.dataset_letter = 'C'
        print(self.dataset)
        prepare_dataset(self)
        prepare_contrast(self)
        self.X_train, self.y_train_df = funkcije.prepare_3_channel_np_arrays(list(set(self.train_df.index.to_list())),
                                                                             self.train_df,
                                                                             self.train_npy,
                                                                             num_of_sices=self.settings.num_of_slices_per_mri,
                                                                             mode=1)
        self.y_train_dummies = funkcije.to_dummies(self.y_train_df, self.settings.modalitete)
        self.analyse_model()

    def train_model(self):
        prepare_dataset(self)
        prepare_contrast(self)
        self.train_idxs, self.val_idxs = funkcije.train_test_split_patient(self.train_df,
                                                                           val_size=0.2,
                                                                           rnd_state=self.settings.SEED_vgg16)
        self.X_train, self.y_train_df = funkcije.prepare_3_channel_np_arrays(self.train_idxs,
                                             self.train_df,
                                             self.train_npy,
                                             num_of_sices=self.settings.num_of_slices_per_mri,
                                             mode = 1)
        self.y_train_dummies = funkcije.to_dummies(self.y_train_df, self.settings.modalitete)
        self.X_val, self.y_val_df = funkcije.prepare_3_channel_np_arrays(self.val_idxs,
                                             self.train_df,
                                             self.train_npy,
                                             num_of_sices=self.settings.num_of_slices_per_mri,
                                             mode=1)
        self.y_val_dummies = funkcije.to_dummies(self.y_val_df, self.settings.modalitete)
        self.model = funkcije.initialize_VGG16(self.settings.image_size, self.y_train_dummies.shape[1])
        self.model = funkcije.train_VGG16(self.model, 
                                          self.X_train, 
                                          self.y_train_dummies, 
                                          self.X_val, 
                                          self.y_val_dummies, 
                                          self.settings.LR_vgg16, 
                                          self.settings.NUM_EPOCHS_vgg16)
        self.analyse_model()

    def analyse_model(self):
        self.y_train_predicted = self.model.predict(self.X_train)
        self.X_test, self.y_test_df = funkcije.prepare_3_channel_np_arrays(list(set(self.y_test_df.index.to_list())),
                                                                             self.y_test_df,
                                                                             self.X_test,
                                                                             num_of_sices=self.settings.num_of_slices_per_mri,
                                                                             mode=1)
        self.y_test_dummies = funkcije.to_dummies(self.y_test_df, self.settings.modalitete)
        self.y_test_predicted = self.model.predict(self.X_test)

        self.score_train = self.model.evaluate(self.X_train, self.y_train_dummies, verbose=0)
        self.score_test = self.model.evaluate(self.X_test, self.y_test_dummies, verbose=0)

        statistical_analysis(self)

        save_results(self)

class eksperimenti:
    def __init__(self, contrast_case_num):
        self.contrast_case_num = contrast_case_num
        if self.contrast_case_num == 0:
            self.contrast_case = 'without_contrast'
        elif self.contrast_case_num == 1:
            self.contrast_case = 'only_t1w_with_contrast'
        elif self.contrast_case_num == 2:
            self.contrast_case = 'all_modals_with_contrast'
    def VGG16(self):
        settings = serialNo_experiment()
        getattr(settings, self.contrast_case)()
        settings.vgg16_parameters()
        ex = pretrained_VGG16(settings)
        ex.dataset_A_experiment()
        if not self.contrast_case_num == 1:
            ex.dataset_B_and_C_experiment()
        # res_img_to_excel(self, ex)

    def MLP(self):
        settings = serialNo_experiment()
        getattr(settings, self.contrast_case)()
        settings.mlp_parameters()
        ex = MLP(settings)
        ex.dataset_A_experiment()
        if not self.contrast_case_num == 1:
            ex.dataset_B_and_C_experiment()
        # res_img_to_excel(self, ex)

    def RF(self):
        settings = serialNo_experiment()
        getattr(settings, self.contrast_case)()
        settings.rf_parameters()
        ex = RF(settings)
        ex.dataset_A_experiment()
        if not self.contrast_case_num == 1:
            ex.dataset_B_and_C_experiment()
        # res_img_to_excel(self, ex)

    def VGG16_and_MLP(self):
        settings = serialNo_experiment()
        getattr(settings, self.contrast_case)()
        settings.vgg16_and_mlp_parameters()
        ex = VGG16_and_MLP(settings)
        ex.dataset_A_experiment()
        if not self.contrast_case_num == 1:
            ex.dataset_B_and_C_experiment()
        # res_img_to_excel(self, ex)

def statistical_analysis(res_class):
        if res_class.dataset_letter == 'B':
            res_class.predicted_b = funkcije.from_dummies(res_class.y_test_predicted, res_class.settings.modalitete)
        elif res_class.dataset_letter == 'C':
            res_class.predicted_c = funkcije.from_dummies(res_class.y_test_predicted, res_class.settings.modalitete)
            res_class.freq_b = np.zeros((len(res_class.settings.modalitete)))
            res_class.freq_c = np.zeros((len(res_class.settings.modalitete)))
            for enum, modal in enumerate(res_class.settings.modalitete):
                res_class.freq_b[enum] = res_class.predicted_b.count(modal)
                res_class.freq_c[enum] = res_class.predicted_c.count(modal)
            res_class.u_test, res_class.p_value = scipy.stats.mannwhitneyu(res_class.freq_b, res_class.freq_c)
        else:
            res_class.u_test = ''
            res_class.p_value = ''

        # ARCHIVE
        if (res_class.settings.contrast_case == 1) or (res_class.settings.contrast_case == 2):
            res_class.archive_modals = ['T1W_NO_CONTRAST' if i == 'T1W' else i for i in res_class.settings.modalitete]
        else:
            res_class.archive_modals = res_class.settings.modalitete
        res_class.archive = pd.DataFrame(data=funkcije.from_dummies(res_class.y_train_predicted, res_class.archive_modals), index=res_class.y_train_df.index, columns=['Predicted'])
        funkcije.archive(res_class.archive)
        res_class.archive = pd.DataFrame(
            data=funkcije.from_dummies(res_class.y_test_predicted, res_class.archive_modals),
            index=res_class.y_test_df.index, columns=['Predicted'])
        funkcije.archive(res_class.archive)
        # AVERAGE
        if res_class.settings.npy_mode == 2:
            res_class.y_train_predicted_df = pd.Series(data=funkcije.from_dummies(res_class.y_train_predicted, res_class.settings.modalitete), index=res_class.y_train_df.index)
            res_class.y_train_average_predicted_df = pd.Series(index=set(res_class.y_train_df.index.to_list()))
            res_class.average_slice_df = pd.DataFrame(index=set(res_class.y_train_df.index.to_list()), columns=res_class.settings.modalitete)
            for idx, row in res_class.average_slice_df.iterrows():
                for modaliteta in res_class.settings.modalitete:
                    res_class.average_slice_df.loc[idx, modaliteta] = res_class.y_train_predicted_df.loc[idx].values.tolist().count(modaliteta)
                if (max(res_class.average_slice_df.loc[idx]) == res_class.average_slice_df.loc[idx]).sum() > 1:
                    res_class.y_train_average_predicted_df.loc[idx] = 0
                else:
                    res_class.y_train_average_predicted_df.loc[idx] = res_class.average_slice_df.loc[idx].astype(np.int).idxmax(axis=1)
            res_class.y_train_average_df = res_class.y_train_df.loc[~res_class.y_train_df.index.duplicated()]
            res_class.y_train_average_predicted_df = res_class.y_train_average_predicted_df.loc[res_class.y_train_average_df.index]
            res_class.y_train_average_dummies = funkcije.to_dummies(res_class.y_train_average_df['sequence'].values, res_class.settings.modalitete)
            res_class.y_train_average_predicted_dummies = funkcije.to_dummies(res_class.y_train_average_predicted_df.values, res_class.settings.modalitete)

            res_class.y_test_predicted_df = pd.Series(
                data=funkcije.from_dummies(res_class.y_test_predicted, res_class.settings.modalitete),
                index=res_class.y_test_df.index)
            res_class.y_test_average_predicted_df = pd.Series(index=set(res_class.y_test_df.index.to_list()))
            res_class.average_slice_df = pd.DataFrame(index=set(res_class.y_test_df.index.to_list()),
                                                      columns=res_class.settings.modalitete)
            for idx, row in res_class.average_slice_df.iterrows():
                for modaliteta in res_class.settings.modalitete:
                    res_class.average_slice_df.loc[idx, modaliteta] = res_class.y_test_predicted_df.loc[
                        idx].values.tolist().count(modaliteta)
                if (max(res_class.average_slice_df.loc[idx]) == res_class.average_slice_df.loc[idx]).sum() > 1:
                    res_class.y_test_average_predicted_df.loc[idx] = 0
                else:
                    res_class.y_test_average_predicted_df.loc[idx] = res_class.average_slice_df.loc[idx].astype(
                        np.int).idxmax(axis=1)
            res_class.y_test_average_df = res_class.y_test_df.loc[~res_class.y_test_df.index.duplicated()]
            res_class.y_test_average_predicted_df = res_class.y_test_average_predicted_df.loc[
                res_class.y_test_average_df.index]
            res_class.y_test_average_dummies = funkcije.to_dummies(res_class.y_test_average_df['sequence'].values,
                                                                    res_class.settings.modalitete)
            res_class.y_test_average_predicted_dummies = funkcije.to_dummies(res_class.y_test_average_predicted_df.values,
                                                                              res_class.settings.modalitete)

def prepare_dataset(dataset_class):
    if dataset_class.dataset_letter == 'A':
        dataset_class.train_npy, dataset_class.train_df = funkcije.load_datasets(dataset_class.settings.npy_base_path,
                                                                             dataset_class.settings.train_file_prefix + dataset_class.dataset)
        dataset_class.X_test, dataset_class.y_test_df = funkcije.load_datasets(dataset_class.settings.npy_base_path,
                                                                              dataset_class.settings.test_file_prefix + dataset_class.dataset)
    elif (dataset_class.dataset_letter == 'B') or (dataset_class.dataset_letter == 'C'):
        dataset_class.train_npy, dataset_class.train_df = funkcije.load_datasets(dataset_class.settings.npy_base_path,
                                                                             'B_and_C_train_zdruzeno')
        if dataset_class.dataset_letter == 'B':
            dataset_class.X_test, dataset_class.y_test_df = funkcije.load_datasets(dataset_class.settings.npy_base_path,
                                                                          'B_test_zdruzeno')
            dataset_class.dataset = 'DATASET_B'
        elif dataset_class.dataset_letter == 'C':
            dataset_class.X_test, dataset_class.y_test_df = funkcije.load_datasets(dataset_class.settings.npy_base_path,
                                                                                      'DATASET_C')
            dataset_class.dataset = 'DATASET_C'

def save_results(train_class):
    final_folder = os.path.basename(train_class.settings.res_path)
    roc_micro, roc_macro = funkcije.saveROC(train_class.y_test_dummies,
                                   train_class.y_test_predicted,
                                   train_class.settings.save_results,
                                   train_class.settings.res_path,
                                   train_class.dataset)
    # roc_micro_average, roc_macro_average = funkcije.saveROC(train_class.y_test_dummies,
    #                                         train_class.y_test_predicted,
    #                                         train_class.settings.save_results,
    #                                         train_class.settings.res_path,
    #                                         train_class.dataset + '_AVERAGE')
    u_test = ''
    p_value = ''
    if train_class.dataset_letter == 'C':
        u_test = str(train_class.u_test)
        p_value = str(train_class.p_value)

    results = {'Input': train_class.dataset,
               'Input_opis': ', '.join(train_class.settings.features),
               'Modalitete': ', '.join(train_class.settings.modalitete),
               'Nastavitve': train_class.settings.nastavitve,
               'Train_acc': train_class.score_train[1],
               'Test_acc': train_class.score_test[1],
               'Train_acc_average': '/', #sklearn.metrics.accuracy_score(train_class.y_train_average_df['sequence'].to_list(), train_class.y_train_average_predicted_df.to_list()),
               'Test_acc_average': '/', #sklearn.metrics.accuracy_score(train_class.y_test_average_df['sequence'].to_list(), train_class.y_test_average_predicted_df.to_list()),
               'Train_loss': train_class.score_train[0],
               'Test_loss': train_class.score_test[0],
               'ROC_micro': roc_micro,
               'ROC_micro_average': '/', #roc_micro_average,
               'U_test': u_test,
               'P_value': p_value}

    funkcije.writeResToExcel(train_class.settings.excel_name,
                    results,
                    train_class.settings.res_path,
                    final_folder)
    funkcije.save_model(train_class.model,
                        str(train_class.dataset) + '_' + train_class.settings.model_name,
                        train_class.settings.res_path)
    funkcije.plot_confusion_matrix(train_class.y_train_dummies,
                                   train_class.y_train_predicted,
                                   train_class.settings.modalitete,
                                   train_class.settings.res_path,
                                   title=str(train_class.dataset) + '_train_cf')
    funkcije.plot_confusion_matrix(train_class.y_test_dummies,
                                   train_class.y_test_predicted,
                                   train_class.settings.modalitete,
                                   train_class.settings.res_path,
                                   title=str(train_class.dataset) + '_test_cf')

def res_img_to_excel(exp_class, train_class):
    if not exp_class.contrast_case_num == 1:
        seznam = ['11018','45321', '70982', '21911', '000000SI4024MR02', '22002', '41597', '141797', '35198', 'DATASET_B', 'DATASET_C']
    else:
        seznam = ['11018', '22002', '141797', '21911', '70982', 'zdruzeno1', 'zdruzeno2']
    for enum, i in enumerate(seznam):
        cf_img = np.array(Image.open(i + '_test_cf.png'))
        roc_img = np.array(Image.open(i + '_ROC.png'))
        if enum == 0:
            final_array = np.concatenate((cf_img, roc_img), axis=1)
        else:
            final_array = np.concatenate(
                (final_array, np.concatenate((cf_img, roc_img), axis=1)),
                axis=0)
    Image.fromarray(final_array).save(os.path.join(train_class.settings.res_path, 'excel_img_res.png'))
    writer = pd.ExcelWriter(os.path.join(train_class.settings.res_path, 'Results.xlsx'), engine='xlsxwriter')
    worksheet = writer.sheets['Sheet1']
    worksheet.insert_image('B25', 'excel_img_res.png')
    writer.save()

def prepare_contrast(experiment_class):
    if experiment_class.settings.contrast_case == 0:
        experiment_class.train_df.true_label[~experiment_class.train_df.true_label.isin(experiment_class.settings.modalitete)] = 'OTHER'
        experiment_class.y_test_df.true_label[~experiment_class.y_test_df.true_label.isin(experiment_class.settings.modalitete)] = 'OTHER'
        experiment_class.y_train_dummies = funkcije.to_dummies(experiment_class.train_df,
                                                               experiment_class.settings.modalitete)
        experiment_class.y_test_dummies = funkcije.to_dummies(experiment_class.y_test_df,
                                                              experiment_class.settings.modalitete)
    elif experiment_class.settings.contrast_case == 1:
        experiment_class.train_df = experiment_class.settings.features_and_references_dataframe.loc[experiment_class.train_df.index].true_label + '_' + experiment_class.settings.features_and_references_dataframe.loc[experiment_class.train_df.index, 'hasContrast (0/1)']
        experiment_class.train_df = experiment_class.train_df.replace(['T1W_0', 'T1W_1'], experiment_class.settings.modalitete).to_frame(name='true_label')
        experiment_class.y_train_dummies = funkcije.to_dummies(experiment_class.train_df['true_label'].to_list(), experiment_class.settings.modalitete)

        experiment_class.y_test_df = experiment_class.settings.features_and_references_dataframe.loc[experiment_class.y_test_df.index].true_label + '_' + experiment_class.settings.features_and_references_dataframe.loc[experiment_class.y_test_df.index, 'hasContrast (0/1)']
        experiment_class.y_test_df = experiment_class.y_test_df.replace(['T1W_0', 'T1W_1'], experiment_class.settings.modalitete).to_frame(name='true_label')
        experiment_class.y_test_dummies = funkcije.to_dummies(experiment_class.y_test_df['true_label'].to_list(), experiment_class.settings.modalitete)

    elif experiment_class.settings.contrast_case == 2:
        modals = experiment_class.settings.features_and_references_dataframe.loc[experiment_class.train_df.index].true_label
        contrasts = experiment_class.settings.features_and_references_dataframe.loc[experiment_class.train_df.index, 'hasContrast (0/1)']
        contrasts[~(contrasts == '1')] = ''
        contrasts[(modals == 'T1W') & (contrasts == '1')] = '_CONTRAST'
        experiment_class.train_df['true_label'] = (modals + contrasts).to_list()
        modals = experiment_class.settings.features_and_references_dataframe.loc[
            experiment_class.y_test_df.index].true_label
        contrasts = experiment_class.settings.features_and_references_dataframe.loc[
            experiment_class.y_test_df.index, 'hasContrast (0/1)']
        contrasts[~(contrasts == '1')] = ''
        contrasts[(modals == 'T1W') & (contrasts == '1')] = '_CONTRAST'
        experiment_class.y_test_df['true_label'] = (modals + contrasts).to_list()
        experiment_class.y_train_dummies = funkcije.to_dummies(experiment_class.train_df,
                                                               experiment_class.settings.modalitete)
        experiment_class.y_test_dummies = funkcije.to_dummies(experiment_class.y_test_df,
                                                              experiment_class.settings.modalitete)
