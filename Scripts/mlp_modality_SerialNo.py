from sklearn.neural_network import MLPClassifier
import glob
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import keras
import pickle
import funkcije
from funkcije import writeResToExcel, saveROC
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout

# contrast = False
# all_modal = False
# b_dataset = False
# c_dataset = False

def ex_mlp(contrast, all_modal, b_dataset, c_dataset):
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
    NUM_EPOCHS = 50
    num_of_neurons_in_hidden_layer = 20
    dropout_rate = 0.2
    actv_fun = 'relu'
    optimizer_algorithm = 'adam'

    # EXCEL DESCRIPTIONS
    save_results = True
    final_folder = 'A_mlp'
    model_name = 'mlp'
    if not contrast:
        pass
    else:
        if not all_modal:
            final_folder = final_folder + '_' + 'CONTRAST'
            model_name = model_name + '_' + 'CONTRAST'
        else:
            final_folder = final_folder + '_' + 'CONTRAST_ALL_MODAL'
            model_name = model_name + '_' + 'CONTRAST_ALL_MODAL'

    nastavitve_CNN = '{} epochs, dropout rate: {}, {} neurons in hidden layer, actv. fun {}, optimizer: {}'.format(NUM_EPOCHS, dropout_rate, num_of_neurons_in_hidden_layer, actv_fun, optimizer_algorithm)
    izbira_rezin = ' '.join(modalitete)
    excel_name = 'MLP_rez'
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

        if not(os.path.isfile(os.path.join(npy_base_path, train_file_name + '.npy'))):
            continue

        _, _, train_df = funkcije.load_datasets(npy_base_path, train_file_name)
        _, _, test_df = funkcije.load_datasets(npy_base_path, test_file_name)

        if c_dataset:
            train_df = pd.concat([train_df, test_df])
            _, _, test_df = funkcije.load_datasets(npy_base_path, 'DATASET_C')
            dataset = 'DATASET_C'
        elif b_dataset:
            _, _, train_df = funkcije.load_datasets(npy_base_path, 'B_train_zdruzeno')
            _, _, test_df = funkcije.load_datasets(npy_base_path, 'B_test_zdruzeno')
            dataset = 'DATASET_B'

        train_tmp = features_and_references_dataframe.loc[train_df.index.to_list()]
        test_tmp = features_and_references_dataframe.loc[test_df.index.to_list()]

        X_train, y_train = funkcije.get_img_params(train_tmp, features, modalitete, modalitete_encoded)
        X_test, y_test = funkcije.get_img_params(test_tmp, features, modalitete, modalitete_encoded)



        if not contrast:
            pass
        else:
            if not all_modal:
                y_train = (features_and_references_dataframe.loc[train_df.index].sequence + '_' +
                           features_and_references_dataframe.loc[train_df.index, 'hasContrast (0/1)']).to_list()
                y_test = (features_and_references_dataframe.loc[test_df.index].sequence + '_' +
                          features_and_references_dataframe.loc[test_df.index, 'hasContrast (0/1)']).to_list()
                y_train = funkcije.to_dummies(y_train, ['T1W_0', 'T1W_1'])
                y_test = funkcije.to_dummies(y_test, ['T1W_0', 'T1W_1'])
            else:
                modals = features_and_references_dataframe.loc[train_df.index].sequence
                contrasts = features_and_references_dataframe.loc[train_df.index, 'hasContrast (0/1)']
                contrasts[~(contrasts == '1')] = ''
                contrasts[(modals == 'T1W') & (contrasts == '1')] = '_CONTRAST'
                y_train = (modals + contrasts).to_list()

                modals = features_and_references_dataframe.loc[test_df.index].sequence
                contrasts = features_and_references_dataframe.loc[test_df.index, 'hasContrast (0/1)']
                contrasts[~(contrasts == '1')] = ''
                contrasts[(modals == 'T1W') & (contrasts == '1')] = '_CONTRAST'
                y_test = (modals + contrasts).to_list()
                y_train = funkcije.to_dummies(y_train, modalitete)
                y_test = funkcije.to_dummies(y_test, modalitete)

        classifier = Sequential()
        # First Hidden Layer
        classifier.add(Dense(num_of_neurons_in_hidden_layer, activation=actv_fun, kernel_initializer='random_normal', input_dim=len(features)))
        classifier.add(Dropout(dropout_rate))
        # Output Layer
        classifier.add(Dense(len(modalitete), activation='sigmoid', kernel_initializer='random_normal'))
        classifier.add(Dropout(dropout_rate))
        # Compiling the neural network
        classifier.compile(optimizer=optimizer_algorithm, loss='binary_crossentropy', metrics=['accuracy'])
        # Fitting the data to the training dataset
        classifier.fit(X_train, y_train, epochs=NUM_EPOCHS)

        y_train_predicted = classifier.predict(X_train)
        y_test_predicted = classifier.predict(X_test)

        score_train = classifier.evaluate(X_train, y_train, verbose=0)
        score_test = classifier.evaluate(X_test, y_test, verbose=0)


        # mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, len(features)), random_state=1)
        # mlp.fit(X_train, y_train)
        # y_test_predicted = mlp.predict(X_test)

        roc_micro, roc_macro = saveROC(y_test, y_test_predicted, save_results, final_folder, dataset)
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
        funkcije.save_model(classifier, str(dataset) + '_' + model_name, res_path)
        funkcije.plot_confusion_matrix(y_train, y_train_predicted, modalitete, res_path,
                                       title=str(dataset) + '_train_cf')
        funkcije.plot_confusion_matrix(y_test, y_test_predicted, modalitete, res_path, title=str(dataset) + '_test_cf')
        if b_dataset or c_dataset:
            return

if __name__ == "__main__":
    print('Nič')