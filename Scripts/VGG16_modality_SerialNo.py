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
import sys

# contrast = False
# all_modal = False
# b_dataset = False
# c_dataset = False

def ex_vgg16(contrast, all_modal, b_dataset, c_dataset):

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
    to_other = ['?', 'SPINE_OTHER', 'SPINE_T1W', 'SPINE_T2W', 'SPINE_FLAIR']

    os.chdir(npy_base_path)

    # MODALITETE ['T1W','T2W','FLAIR','OTHER'], 'T1W_CONTRAST'
    if not contrast:
        modalitete = ['T1W','T2W','FLAIR','OTHER']
    else:
        if not all_modal:
            modalitete = ['T1W', 'T1W_CONTRAST']
        else:
            modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER', 'T1W_CONTRAST']

    save_results = True
    image_size = 128
    SEED = 49
    NUM_EPOCHS = 50
    LR = 1e-4
    nastavitve_CNN = 'VGG16, {} epochs, RMSprop(lr={}), seed={}'.format(NUM_EPOCHS, LR, SEED)
    izbira_rezin = 'Uniformna porazd. 2*(i_dim*0.2, i_dim*0.42), 2*(i_dim*0.58, i_dim*0.8)'
    excel_name = 'vgg16_rez'
    final_folder = 'A_VGG16'
    model_name = 'vgg16'
    if not contrast:
        pass
    else:
        if not all_modal:
            final_folder = final_folder + '_' + 'CONTRAST'
            model_name = model_name + '_' + 'CONTRAST'
        else:
            final_folder = final_folder + '_' + 'CONTRAST_ALL_MODAL'
            model_name = model_name + '_' + 'CONTRAST_ALL_MODAL'
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

    for experiment_num, dataset in enumerate(datasets):
        dataset = dataset[0]
        print('##########     ' + str(experiment_num), end='', flush=True)
        train_file_name = 'A_train' + '_' + dataset
        test_file_name = 'A_test' + '_' + dataset
        if not contrast:
            # CASE: ['T1W','T2W','FLAIR','OTHER']
            pass
        else:
            if not all_modal:
                # CONTRAST CASE: T1W and T1W_CONTRAST
                train_file_name = 'train' + '_' + dataset_names[experiment_num] + '_' + dataset
                test_file_name = 'test' + '_' + dataset_names[experiment_num] + '_' + dataset
            else:
                pass

        if not(os.path.isfile(os.path.join(npy_base_path, train_file_name + '.npy'))):
            continue

        X_train, y_train, y_train_df = funkcije.load_datasets(npy_base_path, train_file_name)
        X_test, y_test, y_test_df = funkcije.load_datasets(npy_base_path, test_file_name)

        if c_dataset:
            X_train = np.concatenate((X_train, X_test), axis=0)
            y_train = y_train + y_test
            y_train_df = pd.concat([y_train_df, y_test_df])
            X_test, y_test, y_test_df = funkcije.load_datasets(npy_base_path, 'DATASET_C')
            dataset = 'DATASET_C'
        elif b_dataset:
            X_train, y_train, y_train_df = funkcije.load_datasets(npy_base_path, 'B_train_zdruzeno')
            X_test, y_test, y_test_df = funkcije.load_datasets(npy_base_path, 'B_test_zdruzeno')
            dataset = 'DATASET_B'

        if not contrast:
            pass
        else:
            if not all_modal:
                y_train = (features_and_references_dataframe.loc[y_train_df.index].sequence + '_' +
                           features_and_references_dataframe.loc[y_train_df.index, 'hasContrast (0/1)']).to_list()
                y_test = (features_and_references_dataframe.loc[y_test_df.index].sequence + '_' +
                          features_and_references_dataframe.loc[y_test_df.index, 'hasContrast (0/1)']).to_list()
            else:

                modals = features_and_references_dataframe.loc[y_train_df.index].sequence
                contrasts = features_and_references_dataframe.loc[y_train_df.index, 'hasContrast (0/1)']
                contrasts[~(contrasts == '1')] = ''
                contrasts[(modals == 'T1W') & (contrasts == '1')] = '_CONTRAST'
                y_train = (modals + contrasts).to_list()

                modals = features_and_references_dataframe.loc[y_test_df.index].sequence
                contrasts = features_and_references_dataframe.loc[y_test_df.index, 'hasContrast (0/1)']
                contrasts[~(contrasts == '1')] = ''
                contrasts[(modals == 'T1W') & (contrasts == '1')] = '_CONTRAST'
                y_test = (modals + contrasts).to_list()

        print('     DATASET:' + dataset + '     ##########')

        if not contrast:
            y_train = ['OTHER' if modal in to_other else modal for modal in y_train]
            y_test = ['OTHER' if modal in to_other else modal for modal in y_test]
        else:
            pass

        if not contrast:
            y_train_dummies = funkcije.to_dummies(y_train, modalitete)
            y_TEST = funkcije.to_dummies(y_test, modalitete)
        else:
            if not all_modal:
                y_train_dummies = funkcije.to_dummies(y_train, ['T1W_0', 'T1W_1'])
                y_TEST = funkcije.to_dummies(y_test, ['T1W_0', 'T1W_1'])
            else:
                y_train_dummies = funkcije.to_dummies(y_train, modalitete)
                y_TEST = funkcije.to_dummies(y_test, modalitete)

        X_TRAIN, X_VALIDATION, y_TRAIN, y_VALIDATION = train_test_split(X_train, y_train_dummies, random_state=SEED)

        model = funkcije.initialize_VGG16(image_size, len(set(y_train)))
        model = funkcije.train_VGG16(model, X_TRAIN, y_TRAIN, X_VALIDATION, y_VALIDATION, LR, NUM_EPOCHS)

        y_train_predicted = model.predict(X_TRAIN)
        y_test_predicted = model.predict(X_test)

        score_train = model.evaluate(X_TRAIN, y_TRAIN, verbose=0)
        score_test = model.evaluate(X_test, y_TEST, verbose=0)

        roc_micro, roc_macro = saveROC(y_TEST, y_test_predicted, save_results, final_folder, dataset)

        if save_results:
            results = {'Input': dataset,
                       'Input_opis': izbira_rezin,
                       'Modalitete': ', '.join(modalitete),
                       'Nastavitve': nastavitve_CNN,
                       'Train_acc': score_train[1],
                       'Test_acc': score_test[1],
                       'Train_loss': score_train[0],
                       'Test_loss': score_test[0],
                       'ROC_micro': roc_micro,
                       'ROC_macro': roc_macro}

            writeResToExcel(excel_name, results, final_folder)

        print('Test loss:', score_test[0])
        print('Test accuracy:', score_test[1])
        funkcije.save_model(model, str(dataset) + '_' + model_name, res_path)
        funkcije.plot_confusion_matrix(y_TRAIN, y_train_predicted, modalitete, res_path, title=str(dataset) + '_train_cf')
        funkcije.plot_confusion_matrix(y_TEST, y_test_predicted, modalitete, res_path, title=str(dataset) + '_test_cf')
        if b_dataset or c_dataset:
            return

if __name__ == "__main__":
    print('Nič')