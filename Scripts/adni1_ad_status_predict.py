import numpy as np
from os.path import exists, join
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
import pickle
from funkcije import resasample_to_size, make_tarfile
from slice_choice_control import get_slices
from matplotlib import pyplot as plt
import random
import funkcije
import keras
from keras.models import Model
from sklearn.model_selection import train_test_split
import scipy
import imp
imp.reload(funkcije)

def prepare_X_y(X_in, y_in_df, indices, mode, num_of_imgs=3, image_size = 128, ref_col_name = 'disease_status'):
    y_out_df = pd.DataFrame()
    if mode == 1 or mode == 3:
        X_out = np.zeros((len(indices), image_size, image_size, num_of_imgs))
        for enum, idx in enumerate(indices):
            slices = np.arange(y_in_df.shape[0])[y_in_df['Image Data ID'] == idx]
            # print(slices)
            X_out[enum] = np.moveaxis(funkcije.standardize(X_in[slices]), 0, -1)
            y_out_df = y_out_df.append(y_in_df.iloc[slices[0]])
    elif mode == 2:
        X_out_i = np.zeros((3, len(indices), image_size, image_size, num_of_imgs))
        list_tmp = y_in_df.index.to_list()
        for iter, idx in enumerate(indices):
            slices = np.arange(y_in_df.shape[0])[y_in_df['Image Data ID'] == idx]
            y_out_df = y_out_df.append(y_in_df.iloc[slices[0]])
            for pos, s in enumerate(slices):
                X_out_i[pos, iter, :, :, :] = np.stack((funkcije.standardize(X_in[s]),) * 3, axis=-1)
        X_out = [X_out_i[0], X_out_i[1], X_out_i[2]]
    return X_out, y_out_df

def cam_random_images(experiment_folder_name, mode, model_name, npy_name):
    base_path = r'/home/jovyan/shared/InteliRad-gasper'
    classes = ['CN', 'MCI', 'AD']
    experiment_path = os.path.join(base_path, experiment_folder_name)
    imgs_folder_path = os.path.join(base_path, 'ADNI_prepared')
    features_and_references_file_name = 'adni_12_1_2020_df'
    features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))
    os.chdir(experiment_path)
    num_of_slices = 3

    if mode == 3:
        model_path = os.path.join(experiment_path, model_name + '.h5')
        final_img_name = 'CAM_random_30'
        rows = 5
        cols = 6
        alpha = 1 - 0.65

        os.chdir(imgs_folder_path)
        tmp_df = pd.read_pickle(npy_name + '_df')
        X_npy = np.load(npy_name + '.npy')
        izbrani = random.sample(set(tmp_df.index.to_list()), 30)
        idxs = []
        for i in izbrani:
            idxs.extend(np.arange(tmp_df.shape[0])[tmp_df.index.isin([i])])
        nepodvojeni = [not (i) for i in tmp_df.loc[izbrani].index.duplicated()]
        idxs = np.array(idxs)[nepodvojeni]
        true_labels = features_and_references_dataframe.loc[izbrani]['disease_status']

        loaded_model = keras.models.load_model(model_path)

        fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
        for i, (idx, seq, izb_name) in enumerate(zip(idxs, true_labels, izbrani)):
            seq = classes[int(seq)]
            imgs_idx = np.arange(tmp_df.shape[0])[tmp_df.index.isin([izb_name])]
            X = np.moveaxis(X_npy[imgs_idx, :, :], 0, -1)
            X_cam, y_pred = create_cam(loaded_model, X, already_3channel=True)

            pred_modal = y_pred[0]
            ax_tmp = ax[i // cols, i % cols]
            ax_tmp.imshow(X_npy[idx, :, :], cmap='gray', aspect='equal', alpha=(1-alpha))

            ax_tmp.imshow(X_cam, cmap='jet', alpha=alpha)

            ax_tmp.set_title('True: ' + seq)
            # ax[i // cols, i % cols].axis('off')
            ax_tmp.set_xlabel('Pred: ' + pred_modal)
            ax_tmp.set_ylabel(izb_name)

            edge_color = 'red'
            if seq.lower() == pred_modal.lower() or pred_modal.lower() in seq.lower():
                edge_color = 'green'

            for spine in ax_tmp.spines.values():
                spine.set_linewidth(3)
                spine.set_edgecolor(edge_color)

        plt.savefig(os.path.join(experiments_path, final_img_name))
        plt.close('all')

def create_cam(loaded_model, X_npy, already_3channel = False):
    gap_weights = loaded_model.layers[-1].get_weights()[0]
    cam_model = Model(inputs=loaded_model.input,
                      outputs=(loaded_model.layers[-3].output, loaded_model.layers[-1].output))
    X_out = np.zeros_like(X_npy)
    y_out = []
    modalitete = ['CN', 'MCI', 'AD']

    if already_3channel:
        if X_npy.ndim == 3:
            features, results = cam_model.predict(X_npy[np.newaxis, :])
        elif X_npy.ndim == 4:
            features, results = cam_model.predict(X_npy)
        pred = np.argmax(results)
        cam_weights = gap_weights[:, pred]
        cam_output = np.dot(features, cam_weights)[0]
        y_out.append(modalitete[pred])

        X_out = scipy.ndimage.zoom(cam_output, int(128 / cam_output.shape[0]), order=3)
    else:
        for slc in range(X_npy.shape[0]):
            X = np.stack((X_npy[slc, :, :],) * 3, axis=-1)
            X = X[np.newaxis, :]

            features, results = cam_model.predict(X)
            pred = np.argmax(results)
            cam_weights = gap_weights[:, pred]
            cam_output = np.dot(features, cam_weights)[0]
            y_out.append(modalitete[pred])

            X_out[slc, :, :] = scipy.ndimage.zoom(cam_output, int(128 / cam_output.shape[0]), order=3)
    return X_out, y_out

adni_folder_path = r'/home/jovyan/shared/mri-brain-prediction/adni'
os.chdir(adni_folder_path)

base_path = '/home/jovyan/shared/InteliRad-gasper'
prepared_folder = 'ADNI_prepared'
npy_path = os.path.join(base_path, prepared_folder)
exp_name = 'ADNI_experiments'
experiments_path = os.path.join(base_path, exp_name)

ref_col_name = 'disease_status'

# npy_name = 'adni_ss_158_12_1_2020'
# npy_name = 'adni_cor2_12_1_2020'
npy_name = 'adni_ss2_12_1_2020'
classes = ['CN', 'MCI', 'AD']

X_all = np.load(os.path.join(npy_path, npy_name + '.npy'))
y_all_df = pd.read_pickle(os.path.join(npy_path, npy_name + '_df'))

num_of_rows = y_all_df.shape[0]
learning_rate = 10e-6
mode = 3

uniq_indices = list(set(y_all_df['Image Data ID']))
idx_train_val, idx_test = train_test_split(uniq_indices, test_size=0.3, random_state=42)
idx_train, idx_val = train_test_split(idx_train_val, test_size=0.15, random_state=42)

X_train, y_train = prepare_X_y(X_all, y_all_df, idx_train, mode=mode)
X_val, y_val = prepare_X_y(X_all, y_all_df, idx_val, mode=mode)
X_test, y_test = prepare_X_y(X_all, y_all_df, idx_test, mode=mode)
y_train_dummies = pd.get_dummies(pd.Categorical(y_train[ref_col_name], categories=[0, 1, 2], ordered=True)).values
y_val_dummies = pd.get_dummies(pd.Categorical(y_val[ref_col_name], categories=[0, 1, 2], ordered=True)).values
y_test_dummies = pd.get_dummies(pd.Categorical(y_test[ref_col_name], categories=[0, 1, 2], ordered=True)).values



model = funkcije.initialize_VGG16(mode, 3)
model, history = funkcije.train_VGG16(model,
                     X_train,
                     y_train_dummies,
                     X_val,
                     y_val_dummies,
                     keras.optimizers.Adam(lr=learning_rate),
                     NUM_EPOCHS=50,
                     BATCH_SIZE=30)
model_name = 'adni' + npy_name
funkcije.save_model(model, model_name, experiments_path)
funkcije.plot_loss_function(history, npy_path, 'adni_test' + npy_name)
y_pred_dummies = model.predict(X_test)
funkcije.plot_confusion_matrix(y_test_dummies, y_pred_dummies, classes, experiments_path, title=npy_name+'_CF')
funkcije.saveROC(y_test_dummies, y_pred_dummies, True, experiments_path, npy_name)
cam_random_images(exp_name, mode=mode, model_name=model_name, npy_name=npy_name)
eval_train = model.evaluate(X_train, y_train_dummies, verbose=0)
eval_test = model.evaluate(X_test, y_test_dummies, verbose=0)

print(eval_train)
print(eval_test)

cn = y_test[ref_col_name] == 0; tip = 'cn'
mci = y_test[ref_col_name] == 1; tip = 'mci'
ad = y_test[ref_col_name] == 2; tip = 'ad'

for skupina, tip in zip([cn, mci, ad], ['cn', 'mci', 'ad']):
    X_cam, y_cam = funkcije.create_cam(model, X_test[skupina], already_3channel=True)

    to_print = np.mean(X_cam/np.sum(np.sum(X_cam, axis=1), axis=0)[np.newaxis, np.newaxis,:], axis=-1)

    img_background = X_train[0,:,:,0]
    alpha = 0.35
    plt.figure()
    plt.imshow(img_background, cmap='gray', aspect='equal', alpha=(1 - alpha))
    plt.imshow(to_print, cmap='jet', alpha=alpha)
    plt.savefig(os.path.join(experiments_path, npy_name+'CAM_' + tip))
    plt.close('all')

    # test
    # for enum, (idx, row) in enumerate(y_all_df.iterrows()):
    #     subj = row['Subject']
    #     mask = y_all_df['Subject'] == subj
    #     tmp = np.array(y_all_df.loc[mask, ref_col_name].to_list())
    #     if (tmp[0] == tmp).all():
    #         pass
    #     else:
    #         print(subj)