import numpy as np
from funkcije import writeResToExcel, saveROC
from sklearn.neural_network import MLPClassifier
import glob
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
import funkcije
import pickle
import keras
from keras import models
from keras import optimizers
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout, Input
import pydot

# contrast = False
# all_modal = False
# b_dataset = False
# c_dataset = False

def ex_vgg_mlp(contrast, all_modal, b_dataset, c_dataset):
    # PATHS
    base_path = r'/home/jovyan/shared/InteliRad-gasper'
    img_base_path = os.path.join(base_path, 'images')

    if b_dataset and c_dataset:
        print('Ne smeta biti oba')
        sys.exit()
    npy_base_path = os.path.join(base_path, 'PREPARED_IMAGES')
    if not contrast:
        # CASE: ['T1W','T2W','FLAIR','OTHER']
        pass
    else:
        if not all_modal:
            # CONTRAST CASE: T1W and T1W_CONTRAST
            npy_base_path = os.path.join(base_path, 'PREPARED_IMAGES_CONTRAST')
        else:
            pass
    results_folder = '/home/jovyan/shared/InteliRad-gasper/REZULTATI/'
    os.chdir(npy_base_path)

    # EXPERIMENT SETTINGS
    to_other = ['?', 'SPINE_OTHER', 'SPINE_T1W', 'SPINE_T2W', 'SPINE_FLAIR']
    if not contrast:
        modalitete = ['T1W','T2W','FLAIR','OTHER']
    else:
        if not all_modal:
            modalitete = ['T1W', 'T1W_CONTRAST']
        else:
            modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER', 'T1W_CONTRAST']
    modalitete_encoded = np.arange(len(modalitete)).astype(int)
    features = ['RepetitionTime', 'EchoTime', 'InversionTime', 'MagneticFieldStrength', 'SliceThickness', 'NumberofAverages', 'ImagingFrequency', 'NumberofPhaseEncodingSteps',
                'EchoTrainLength', 'PercentSampling', 'PercentPhaseFieldofView', 'PixelBandwidth', 'FlipAngle']

    # MULTILAYER PERCEPTRON SETTINGS
    NUM_EPOCHS_mlp = 50
    num_of_neurons_in_hidden_layer = 20
    dropout_rate = 0.2
    actv_fun = 'relu'
    optimizer_algorithm = 'adam'
    nastavitve_MLP = '{} epochs, dropout rate: {}, {} neurons in hidden layer, actv. fun {}, optimizer: {}'.format(NUM_EPOCHS_mlp, dropout_rate, num_of_neurons_in_hidden_layer, actv_fun, optimizer_algorithm)

    # CNN SETTINGS
    image_size = 128
    SEED = 49
    NUM_EPOCHS = 50
    LR = 1e-4
    nastavitve_CNN = 'VGG16, {} epochs, RMSprop(lr={}), seed={}'.format(NUM_EPOCHS, LR, SEED)

    # EXCEL DESCRIPTIONS
    save_results = True
    final_folder = 'A_VGG16&mlp_experiment'
    model_name = 'vgg&mlp'
    if not contrast:
        pass
    else:
        if not all_modal:
            final_folder = final_folder + '_' + 'CONTRAST'
            model_name = model_name + '_' + 'CONTRAST'
        else:
            final_folder = final_folder + '_' + 'CONTRAST_ALL_MODAL'
            model_name = model_name + '_' + 'CONTRAST_ALL_MODAL'
    izbira_rezin = ' '.join(modalitete)
    excel_name = 'Rez'
    if b_dataset:
        excel_name = excel_name + '_B'
        model_name = model_name + '_B'
    elif c_dataset:
        excel_name = excel_name + '_C'
        model_name = model_name + '_C'
    res_path = os.path.join(results_folder, final_folder)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # Dataframe import
    features_and_references_file_name = 'features_and_references_dataframe_1866'
    features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))
    features_and_references_dataframe['sequence'] = features_and_references_dataframe['sequence'].mask(~features_and_references_dataframe.sequence.isin(modalitete), 'OTHER')

    datasets = [['11018'],['45321'], ['70982'], ['21911'], ['000000SI4024MR02'], ['22002'], ['41597'], ['141797'], ['35198']]
    if not contrast:
        # CASE: ['T1W','T2W','FLAIR','OTHER']
        pass
    else:
        if not all_modal:
            # CONTRAST CASE: T1W and T1W_CONTRAST
            datasets = [['11018'], ['22002'], ['141797'], ['21911'], ['70982'], ['zdruzeno'], ['zdruzeno']]
            dataset_names = ['A', 'A', 'A', 'A', 'A', 'B1', 'B2']
        else:
            pass

    for num, dataset in enumerate(datasets):
        dataset = dataset[0]
        print('###############     ' + str(num) + '     ###############')

        #### DATA IMPORT
        # Import data for CNN
        train_file_name = 'A_train' + '_' + dataset
        test_file_name = 'A_test' + '_' + dataset
        if not contrast:
            # CASE: ['T1W','T2W','FLAIR','OTHER']
            pass
        else:
            if not all_modal:
                # CONTRAST CASE: T1W and T1W_CONTRAST
                train_file_name = 'train' + '_' + dataset_names[num] + '_' + dataset
                test_file_name = 'test' + '_' + dataset_names[num] + '_' + dataset
            else:
                pass

        if not (os.path.isfile(os.path.join(npy_base_path, train_file_name + '.npy'))):
            continue
        X_train_cnn, y_train_cnn, y_train_cnn_df = funkcije.load_datasets(npy_base_path, train_file_name)
        X_TEST_cnn, y_test_cnn, y_test_cnn_df = funkcije.load_datasets(npy_base_path, test_file_name)

        if c_dataset:
            X_train = np.concatenate((X_train_cnn, X_TEST_cnn), axis=0)
            y_train = y_train_cnn + y_test_cnn
            y_train_df = pd.concat([y_train_cnn_df, y_test_cnn_df])
            X_TEST_cnn, y_test_cnn, y_test_cnn_df = funkcije.load_datasets(npy_base_path, 'DATASET_C')
            dataset = 'DATASET_C'
        elif b_dataset:
            X_train_cnn, y_train_cnn, y_train_cnn_df = funkcije.load_datasets(npy_base_path, 'B_train_zdruzeno')
            X_TEST_cnn, y_test_cnn, y_test_cnn_df = funkcije.load_datasets(npy_base_path, 'B_test_zdruzeno')
            dataset = 'DATASET_B'

        y_train_cnn = funkcije.to_dummies(['OTHER' if modal in to_other else modal for modal in y_train_cnn], modalitete)
        y_TEST_cnn = funkcije.to_dummies(['OTHER' if modal in to_other else modal for modal in y_test_cnn], modalitete)

        # Import data for mlp model
        train_tmp = features_and_references_dataframe.loc[y_train_cnn_df.index.to_list()]
        test_tmp = features_and_references_dataframe.loc[y_test_cnn_df.index.to_list()]

        X_train_mlp, y_train_mlp = funkcije.get_img_params(train_tmp.loc[list(set(y_train_cnn_df.index.to_list()))], features, modalitete, modalitete_encoded)
        X_TEST_mlp, y_TEST_mlp = funkcije.get_img_params(test_tmp.loc[list(set(y_test_cnn_df.index.to_list()))], features, modalitete, modalitete_encoded)

        if not contrast:
            pass
        else:
            if not all_modal:
                y_train = (features_and_references_dataframe.loc[y_train_cnn_df.index].sequence + '_' +
                           features_and_references_dataframe.loc[y_train_cnn_df.index, 'hasContrast (0/1)']).to_list()
                y_test = (features_and_references_dataframe.loc[y_test_cnn_df.index].sequence + '_' +
                          features_and_references_dataframe.loc[y_test_cnn_df.index, 'hasContrast (0/1)']).to_list()
                y_train_cnn = funkcije.to_dummies(y_train, ['T1W_0', 'T1W_1'])
                y_TEST_cnn = funkcije.to_dummies(y_test, ['T1W_0', 'T1W_1'])
            else:
                modals = features_and_references_dataframe.loc[y_train_cnn_df.index].sequence
                contrasts = features_and_references_dataframe.loc[y_train_cnn_df.index, 'hasContrast (0/1)']
                contrasts[~(contrasts == '1')] = ''
                contrasts[(modals == 'T1W') & (contrasts == '1')] = '_CONTRAST'
                y_train = (modals + contrasts).to_list()

                modals = features_and_references_dataframe.loc[y_test_cnn_df.index].sequence
                contrasts = features_and_references_dataframe.loc[y_test_cnn_df.index, 'hasContrast (0/1)']
                contrasts[~(contrasts == '1')] = ''
                contrasts[(modals == 'T1W') & (contrasts == '1')] = '_CONTRAST'
                y_test = (modals + contrasts).to_list()
                y_train_cnn = funkcije.to_dummies(y_train, modalitete)
                y_TEST_cnn = funkcije.to_dummies(y_test, modalitete)


        X_TRAIN_cnn, X_VALIDATION_cnn, y_TRAIN_cnn, y_VALIDATION_cnn, X_TRAIN_mlp, X_VALIDATION_mlp, y_TRAIN_mlp, y_VALIDATION_mlp = train_test_split(X_train_cnn, y_train_cnn, X_train_mlp, y_train_mlp,
                                                                                        random_state=SEED)
        #### MODEL
        # Model architecture
        mlp_inputs = Input(shape=(X_TRAIN_mlp.shape[1],))
        x = Dense(num_of_neurons_in_hidden_layer, activation=actv_fun)(mlp_inputs)
        x = Dropout(rate=dropout_rate)(x)
        # x = Dense(num_of_neurons_in_hidden_layer, activation='sigmoid', kernel_initializer='random_normal')(x)
        # x = Dropout(rate=dropout_rate)(x)
        mlp_out = Dense(len(modalitete), activation='sigmoid', kernel_initializer='random_normal')(x)

        vgg_cnn = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(image_size, image_size, 3))

        # Freeze the layers except the last 4 layers
        for layer in vgg_cnn.layers[:-4]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in vgg_cnn.layers:
            print(layer, layer.trainable)

        vgg_cnn.summary()

        cnn_inputs = Input(shape=X_TRAIN_cnn.shape[1:],)

        vgg_out = vgg_cnn(cnn_inputs)

        vgg_flattened = Flatten()(vgg_out)
        merged = keras.layers.concatenate(inputs=[vgg_flattened, mlp_out], axis=1)
        combined = Dense(50, activation='sigmoid')(merged)
        combined = Dropout(rate=dropout_rate)(combined)
        final_layer = Dense(len(modalitete), activation='relu')(combined)

        merged_model = Model(inputs=[cnn_inputs, mlp_inputs], outputs=final_layer)
        merged_model.summary()

        os.chdir(res_path)
        keras.utils.plot_model(merged_model, to_file='merged_model.png')

        merged_model.compile(loss='categorical_crossentropy',
                             optimizer=optimizers.RMSprop(lr=LR),
                             metrics=['acc'])

        merged_model.fit([X_TRAIN_cnn, X_TRAIN_mlp],
                         y=y_TRAIN_cnn,
                         epochs=NUM_EPOCHS,
                         verbose=1,
                         shuffle=True,
                         validation_data=([X_VALIDATION_cnn, X_VALIDATION_mlp], y_VALIDATION_cnn))

        score_train = merged_model.evaluate([X_TRAIN_cnn, X_TRAIN_mlp], y_TRAIN_cnn, verbose=0)
        score_test = merged_model.evaluate([X_TEST_cnn, X_TEST_mlp], y_TEST_cnn, verbose=0)

        print('Train loss: {}, train accuracy: {}'.format(score_train[0], score_train[1]))
        print('Test loss: {}, test accuracy: {}'.format(score_test[0], score_test[1]))

        y_train_predicted = merged_model.predict([X_TRAIN_cnn, X_TRAIN_mlp])
        y_test_predicted = merged_model.predict([X_TEST_cnn, X_TEST_mlp])

        roc_micro, roc_macro = saveROC(y_TEST_cnn, y_test_predicted, save_results, final_folder, dataset)
        results = {'Input': dataset,
                   'Input_opis': ', '.join(features),
                   'Modalitete': ', '.join(modalitete),
                   'Nastavitve': nastavitve_CNN,
                   'Train_acc': score_train[1],
                   'Test_acc': score_test[1],
                   'Train_loss': score_train[0],
                   'Test_loss': score_test[0],
                   'ROC_micro': roc_micro,
                   'ROC_macro': roc_macro}

        writeResToExcel(excel_name, results, final_folder)
        funkcije.save_model(merged_model, str(dataset) + '_' + model_name, res_path)
        funkcije.plot_confusion_matrix(y_TRAIN_cnn, y_train_predicted, modalitete, res_path,
                                       title=str(dataset) + '_train_cf')
        funkcije.plot_confusion_matrix(y_TEST_cnn, y_test_predicted, modalitete, res_path, title=str(dataset) + '_test_cf')
        if b_dataset or c_dataset:
            return

if __name__ == "__main__":
    print('Nič')