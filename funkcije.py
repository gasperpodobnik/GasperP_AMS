from __future__ import print_function
from keras.applications.vgg16 import VGG16
# import matplotlib.pyplot as plt
import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model
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
from sklearn import metrics
import itertools
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import tarfile
import scipy
import random
import shutil
from keras.utils import plot_model
from contextlib import redirect_stdout

def train(X_train, X_test, y_train, y_test):
    # define constants
    NUM_CLASSES = 2
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # initialize sequential model
    model = Sequential()
    # add 1st conv layer: 3x3, relu activation
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=X_train.shape[1:]))
    # 2nd conv layer: 3x3, relu activation
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # max pooling 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    # fully-connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # povzetek strukture modela in števila parametrov
    # write model summary to console
    model.summary()

    # compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=LEARNING_RATE), # Adadelta, RMSprop, SGD,...
                  metrics=['accuracy'])

    run_count = 0
    while exists('./graphs/' + str(run_count)):
        run_count += 1

    tbCallBack = keras.callbacks.TensorBoard(
        log_dir='./graphs/' + str(run_count),
        histogram_freq=0,
        write_graph=True,
        write_images=True)

    class TestCallback(Callback):
        def __init__(self, test_data):
            self.test_data = test_data

        def on_epoch_end(self, epoch, logs={}):
            x, y = self.test_data
            loss, acc = self.model.evaluate(x, y, verbose=0)
            print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

    # zaženi učenje modela
    # TRAIN MODEL
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[tbCallBack, TestCallback((X_test, y_test))])
    score = model.evaluate(X_train, y_train, verbose=0)

    # print training data loo and accuracy
    print('Učna zbirka')
    print('\tloss:', score[0])
    print('\taccuracy:', score[1])
    score = model.evaluate(X_test, y_test, verbose=0) # evaluate model with test dataset
    # print test data loo and accuracy
    print('Testna zbirka')
    print('\tloss:', score[0])
    print('\taccuracy:', score[1])

    return model

def resasample_to_size(input_image, output_size=(128,128), inter_type=sitk.sitkLinear):
    """
    Resample image to desired output size

    :param input_image: Image to resample
    :param output_size: Size of the output image
    :param output_size: Size of the output image

    :return: image of the output size
    """

    spacing = np.array(input_image.GetSize())/np.array(output_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(inter_type)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(output_size)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())

    resampled_image = resampler.Execute(input_image)

    return resampled_image

def resample_image_ams(input_image, spacing_mm=(1, 1, 1), spacing_image=None, inter_type=sitk.sitkLinear):
    """
    Resample image to desired pixel spacing.

    Should specify destination spacing immediate value in parameter spacing_mm or as SimpleITK.Image in spacing_image.
    You must specify either spacing_mm or spacing_image, not both at the same time.

    :param input_image: Image to resample.
    :param spacing_mm: Spacing for resampling in mm given as tuple or list of two/three (2D/3D) float values.
    :param spacing_image: Spacing for resampling taken from the given SimpleITK.Image.
    :param inter_type: Interpolation type using one of the following options:
                            SimpleITK.sitkNearestNeighbor,
                            SimpleITK.sitkLinear,
                            SimpleITK.sitkBSpline,
                            SimpleITK.sitkGaussian,
                            SimpleITK.sitkLabelGaussian,
                            SimpleITK.sitkHammingWindowedSinc,
                            SimpleITK.sitkBlackmanWindowedSinc,
                            SimpleITK.sitkCosineWindowedSinc,
                            SimpleITK.sitkWelchWindowedSinc,
                            SimpleITK.sitkLanczosWindowedSinc
    :type input_image: SimpleITK.Image
    :type spacing_mm: Tuple[float]
    :type spacing_image: SimpleITK.Image
    :type inter_type: int
    :rtype: SimpleITK.Image
    :return: Resampled image as SimpleITK.Image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(inter_type)

    if (spacing_mm is None and spacing_image is None) or \
       (spacing_mm is not None and spacing_image is not None):
        raise ValueError('You must specify either spacing_mm or spacing_image, not both at the same time.')

    if spacing_image is not None:
        spacing_mm = spacing_image.GetSpacing()

    input_spacing = input_image.GetSpacing()
    # set desired spacing
    resampler.SetOutputSpacing(spacing_mm)
    # compute and set output size
    output_size = np.array(input_image.GetSize()) * np.array(input_spacing) \
                  / np.array(spacing_mm)
    output_size = list((output_size + 0.5).astype('uint32'))
    output_size = [int(size) for size in output_size]
    resampler.SetSize(output_size)

    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())

    resampled_image = resampler.Execute(input_image)

    return resampled_image

def standardize(img_np):
    img_np = np.asarray(img_np, dtype=np.float)
    img_np -= np.min(img_np).astype(np.float)
    img_np /= np.max(img_np).astype(np.float)

    return img_np

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def saveImage(img_np, name):
    base_path = r'/home/jovyan/shared/InteliRad-gasper'
    img_np = standardize(img_np)*255
    os.chdir(os.path.join(base_path, 'ogledSlik'))
    im = Image.fromarray(img_np).convert('RGB').convert("P")
    im.save(name + '.png')
    return print('Image saved!')

def writeResToExcel(excel_name, new_data, results_folder, final_folder):
    os.chdir(results_folder)
    file_name = excel_name + '.xlsx'
    if not(os.path.isfile(file_name)):
        writer = pd.ExcelWriter(file_name)#, engine='xlsxwriter')
        header = ['Eksperiment', 'Input', 'Train_acc', 'Train_acc_average', 'Test_acc', 'Test_acc_average', 'Train_loss', 'Test_loss', 'ROC_micro', 'ROC_micro_average', 'U_test', 'P_value']
        pd.DataFrame([[' '],
                      [' ', 'Folder with ROCs: ' + final_folder],
                      [' ', new_data['Input_opis']],
                      [' ', new_data['Modalitete']],
                      [' ', new_data['Nastavitve']],
                      header]).to_excel(writer, sheet_name='Sheet1', header=False, index=False)
        writer.save()
    content = pd.read_excel(file_name)
    col_names = content.columns.values
    cont_np = content.values
    cont_np = np.concatenate(([col_names], cont_np))
    if not(cont_np[-1,0].isdigit()):
        exp_num = 1
    else:
        exp_num = int(cont_np[-1,0])+1
    new_np = np.array([[exp_num,
                  new_data['Input'],
                  new_data['Train_acc'],
                  new_data['Train_acc_average'],
                  new_data['Test_acc'],
                  new_data['Test_acc_average'],
                  new_data['Train_loss'],
                  new_data['Test_loss'],
                  new_data['ROC_micro'],
                  new_data['ROC_micro_average'],
                  new_data['U_test'],
                  new_data['P_value']
                        ]])
    np_to_write = np.concatenate((cont_np, new_np))
    to_write = pd.DataFrame(np_to_write)
    writer = pd.ExcelWriter(file_name)
    to_write.to_excel(writer, index=False, header=None)
    writer.save()
    file = open('exp_num.txt', 'w')
    file.write(str(exp_num))
    file.close()

def writeResToExcel_from_df(excel_name, meta_data, new_data_df, header, results_folder, final_folder):
    os.chdir(results_folder)
    file_name = excel_name + '.xlsx'
    writer = pd.ExcelWriter(file_name)
    for d in meta_data:
        row = [''] * new_data_df.shape[1]
        row[0] = str(d)
        new_data_df = new_data_df.append(pd.DataFrame([row], columns=header))
    new_data_df.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
    writer.save()

def saveROC(y_test, y_pred, save_figure, res_path, graph_num):
    fig_name = str(graph_num) + '_ROC'
    n_classes = y_pred.shape[1]
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    if save_figure:
        fig = plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(res_path, fig_name + '.png'))
        plt.close(fig)
    return roc_auc['micro'], roc_auc['macro']

def initialize_VGG16(mode, num_of_outputs, image_size = 128):
    if mode == 1:  # three different slices in three channels of image
        vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

        # Freeze the layers except the last 4 layers
        for layer in vgg_conv.layers[:-4]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        # for layer in vgg_conv.layers:
        #     print(layer, layer.trainable)

        vgg_conv.summary()

        # Create the model
        model = models.Sequential()

        # Add the vgg convolutional base model
        model.add(vgg_conv)

        # Add new layers
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_of_outputs, activation='softmax'))

    elif mode == 2:  # three same images in three channels
        input1 = Input(shape=(image_size, image_size, 3))
        input2 = Input(shape=(image_size, image_size, 3))
        input3 = Input(shape=(image_size, image_size, 3))
        vgg_conv = VGG16(weights='imagenet', include_top=False)
        # Freeze the layers except the last 4 layers
        for layer in vgg_conv.layers[:-4]:
            layer.trainable = False

        after_vgg1 = vgg_conv(input1)
        after_vgg2 = vgg_conv(input2)
        after_vgg3 = vgg_conv(input3)

        merged = keras.layers.concatenate([Flatten()(after_vgg1), Flatten()(after_vgg2), Flatten()(after_vgg3)],
                                          axis=-1)
        merged = Dense(256, activation='relu')(merged)
        merged = Dropout(0.5)(merged)
        merged = Dense(num_of_outputs, activation='softmax')(merged)
        model = Model(inputs=[input1, input2, input3], outputs=merged)

    elif mode == 3:  # three different slices in three channels of image, CAM!
        vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

        # Freeze the layers except the last 4 layers
        for layer in vgg_conv.layers[:-4]:
            layer.trainable = False
        vgg_conv.layers.pop()
        vgg_conv.layers.pop()
        vgg_conv.layers.pop()
        vgg_conv.layers.pop()
        vgg_conv.layers.pop()
        # vgg_conv.outputs = [vgg_conv.layers[-1].output]
        # vgg_conv.output_layers = [vgg_conv.layers[-1]]
        # vgg_conv.layers[-1].outbound_nodes = []

        inp = vgg_conv.input
        out = vgg_conv.layers[-1].output

        model0 = Model(inp, out)  # create a new model which doesn't have the last two layers in VGG16
        model0.summary()

        # Create the model
        model = models.Sequential()

        # Add the vgg convolutional base model
        model.add(model0)

        # Add new layers
        model.add(layers.Conv2D(1024, 3, padding='same', activation='relu'))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(num_of_outputs, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()
    return model

def train_VGG16(model, X_train, y_train, X_val, y_val, opt_and_lr, NUM_EPOCHS, BATCH_SIZE, loss='categorical_crossentropy'):
    model.compile(loss=loss,
                  optimizer=opt_and_lr,
                  metrics=['acc'])
    history = model.fit(X_train, y_train,
              epochs=NUM_EPOCHS,
              validation_data=(X_val, y_val),
              batch_size=BATCH_SIZE)

    return model, history

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_dummies(y_df, categ):
    return pd.get_dummies(pd.Categorical(y_df.iloc[:,0], categories=categ, ordered=True)).values

def from_dummies(y, categ):
    y_df = pd.DataFrame(y)
    y_df.columns = categ
    return y_df.idxmax(axis=1).to_list()

def load_datasets(base_path, file_name, out_classes):
    features_and_references_file_name = 'Data/features_and_references_dataframe_1866'
    features_and_references_dataframe = pd.read_pickle(os.path.join(r'Data', features_and_references_file_name))
    if len(out_classes) == 4:
        col = 'sequence'
    elif len(out_classes) == 5:
        col = 'sequence_contrast'
    X_out = np.load(os.path.join(base_path, file_name + '.npy'))
    y_out_df = pd.read_pickle(os.path.join(base_path, file_name + '_df'))
    true_labels = features_and_references_dataframe.loc[y_out_df.index, col]
    y_out_df['true_label'] = true_labels
    return X_out, to_other(y_out_df, out_classes)

def to_other(df, out_classes):
    df.true_label[~df.true_label.isin(out_classes)] = 'OTHER'
    return df

def get_img_params(df, features, modality, modal_code):
    y_name = 'sequence'
    params_df = df[features]
    params_df = params_df.mask(params_df == 'N-A', -1).apply(pd.to_numeric)
    y_df = pd.DataFrame(df[y_name])
    # for i, m in enumerate(modality):
    #     y_df = y_df.mask(y_df == m, modal_code[i])
    # one_hot_labels = keras.utils.to_categorical(y_df.values.reshape((y_df.shape[0],1)), num_classes=len(modality))
    return params_df.values, y_df

def plot_confusion_matrix(y_true, y_pred, classes, location,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = np.asarray(classes)
    y_true = from_dummies(y_true, range(len(classes)))
    y_pred = from_dummies(y_pred, range(len(classes)))
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(location, title + '.png'), dpi=fig.dpi)
    plt.close(fig)

def plot_loss_function(history, location, title):
    fig, ax = plt.subplots(2, 1, squeeze=True, constrained_layout=True)
    ax[0].plot(history.history['acc'])
    ax[0].plot(history.history['val_acc'])
    ax[0].set_title('Model accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Val'], loc='lower right')
    # summarize history for loss
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Val'], loc='upper right')
    fig.savefig(os.path.join(location, title + '.png'), dpi=fig.dpi)
    plt.close(fig)

def train_test_split_patient(y_df, experiment_folder_path=None, val_size=0.2, rnd_state = 49):
    imgs_names = list(set(y_df.index.to_list()))
    train_names, val_names = train_test_split(imgs_names, test_size=val_size, random_state=rnd_state)
    if experiment_folder_path is not None:
        with open(os.path.join(experiment_folder_path, "train_names.txt"), "wb") as fp:  # Pickling
            pickle.dump(train_names, fp)
        with open(os.path.join(experiment_folder_path, "val_names.txt"), "wb") as fp:  # Pickling
            pickle.dump(val_names, fp)
    return train_names, val_names

def prepare_3_channel_np_arrays(idx_names, y_df, npy, num_of_sices, mode = 1):
    '''
    :param idx_names: List of non-duplicated indices from reference dataframe
    :param y_df: Reference dataframe. If idx_names list is for training/valiation
        dataset, then y_df should be for training set, otherwise for test set
    :param npy: file with slices from MRI images
    :param num_of_sices: number of slice taken from same MRI image
    :param mode: What to write in three channel image. If mode = 1: three different
        slices in three channels of image, if mode = 2 three same images in three channels
    :return: prepared ndarray for training, reference dataframe
    '''
    img_size = np.asarray(npy.shape[1:])
    df_out = pd.DataFrame(columns=['true_label'])
    n_of_channels = 3 # number of channels in output image (for RGB image is 3)
    if mode == 1 or mode == 3: # three different slices in three channels of image
        out_npy = np.empty((len(idx_names), img_size[0], img_size[1], n_of_channels))
        for iter, idx in enumerate(idx_names):
            indices = [i for i, x in enumerate(y_df.index.to_list()) if x == idx]
            out_npy[iter, :, :, :] = np.moveaxis(npy[indices,:,:], 0, -1)
            df_out = df_out.append(pd.DataFrame(data=[y_df['true_label'].iloc[indices[0]]],
                                       columns=['true_label'],
                                       index=[idx]))
    elif mode == 2: # three same images in three channels
        list_tmp = y_df.index.to_list()
        out_npy = np.empty((num_of_sices, len(idx_names), img_size[0], img_size[1], n_of_channels))
        for iter, idx in enumerate(idx_names):
            df_out = df_out.append(pd.DataFrame(data=[y_df['true_label'].iloc[list_tmp.index(idx)]],
                                                columns=['true_label'],
                                                index=[idx]))
            for s in range(num_of_sices):
                num = list_tmp.index(idx)
                out_npy[s, iter, :, :, :] = np.stack((npy[num,:,:],) * n_of_channels, axis=-1)
                list_tmp[num] = None

    return out_npy, df_out

def save_model(model, name, folder):
    # m_type = name.split('_')[1]
    pth = os.path.join(folder,name)
    if 'rf' in name.lower():
        pickle.dump(model, open(pth + '.sav', 'wb'))
    else:
        model.save(pth + '.h5')

def archive(input_df):
    base_path = r'/home/jovyan/shared/InteliRad-gasper'
    archive_df = pd.read_pickle(os.path.join(base_path,'archive_df'))
    tmp = to_dummies(pd.DataFrame([i[0] for i in input_df.values]), archive_df.columns[:-1])
    ena = archive_df.loc[input_df.index, archive_df.columns[:-1]].values
    ena[tmp.astype(bool)] += 1
    archive_df.loc[input_df.index, archive_df.columns[:-1]] = ena
    archive_df.to_pickle(os.path.join(base_path, 'archive_df'))

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def make_zipfile(output_filename, dir_name):
    shutil.make_archive(output_filename, 'zip', dir_name)

def each_set_preprocess_for_modalities(mode, list_of_names, df, npy, num_of_slices_per_mri, modalitete):
    X, y_df = prepare_3_channel_np_arrays(list_of_names,
                                                   df,
                                                   npy,
                                                   num_of_sices=num_of_slices_per_mri,
                                                   mode=mode)
    if mode == 2:  # three same images in three channels
        X = [X[i] for i in range(num_of_slices_per_mri)]

    y_dummies = to_dummies(y_df, modalitete)
    return X, y_df, y_dummies

# Function to insert row in the dataframe
def Insert_row(row_number, df, row_value):
    # Starting value of upper half
    start_upper = 0

    # End value of upper half
    end_upper = row_number

    # Start value of lower half
    start_lower = row_number

    # End value of lower half
    end_lower = df.shape[0]

    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]

    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]

    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]

    # Combine the two lists
    index_ = upper_half + lower_half

    # Update the index of the dataframe
    df.index = index_

    # Insert a row at the end
    df.loc[row_number] = row_value

    # Sort the index labels
    df = df.sort_index()

    # return the dataframe
    return df

def izris_nap_klasif(experiment_folder_name, imgs_folder_name):
    base_path = r'/home/jovyan/shared/InteliRad-gasper'
    experiment_path = os.path.join(base_path, experiment_folder_name)
    imgs_folder_path = os.path.join(base_path, imgs_folder_name)
    features_and_references_file_name = 'Data/features_and_references_dataframe_1866'
    features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))
    os.chdir(experiment_path)
    num_of_slices = 3
    device_serial_numbers = ['11018', '45321', '70982', '21911', '000000SI4024MR02', '22002', '41597', '141797',
                             '35198', 'B', 'C']
    res_folders = [filename for filename in os.listdir('.') if filename.startswith("RES")]
    for folder in res_folders:
        folder_path = os.path.join(experiment_path, folder)
        for dataset_num in device_serial_numbers:
            curr_folder = os.path.join(folder_path, dataset_num)
            os.chdir(curr_folder)
            for t in ['train', 'test']:
                df_names = [filename for filename in os.listdir(curr_folder) if filename.endswith("_" + t.upper() + "napacno_klasificirani_df")]
                model_names = [filename for filename in os.listdir(curr_folder) if
                            filename.endswith(".h5")]
                for df_name in df_names:
                    df = pd.read_pickle(df_name)
                    file_name = df_name.split('_')
                    ser_num = file_name[-4]
                    mode = int(folder.split('_')[-2])
                    if dataset_num != 'C' and mode == 3:
                        loaded_model = keras.models.load_model(model_names[0])
                    if mode == 3:
                        alpha = 1 - 0.65
                    else:
                        alpha = 0
                    df_ref = pd.read_pickle(os.path.join(imgs_folder_path, t + '_' + ser_num + '_df'))
                    X_npy = np.load(os.path.join(imgs_folder_path, t + '_' + ser_num + '.npy'))
                    rows = df.shape[0]
                    cols = num_of_slices
                    # fig, ax = plt.subplots(rows, cols, figsize=(cols*3, rows*3), constrained_layout=True) #figsize=(cols*3, rows*3),
                    fig = plt.figure(figsize=(cols*3, rows*3))
                    for enum, idx in enumerate(df.index):
                        imgs_idx = np.arange(df_ref.shape[0])[df_ref.index.isin([idx])]
                        true_label = features_and_references_dataframe.loc[idx]['sequence_contrast']
                        pred_label = df.loc[idx, 'pred_label']
                        if mode == 3:
                            X = np.moveaxis(X_npy[imgs_idx, :, :], 0, -1)
                            X_cam, _ = create_cam(loaded_model, X, already_3channel=True)
                        for enum_s, slice in enumerate(imgs_idx):
                            plt.subplot(rows, cols, enum*cols+enum_s+1)
                            plt.imshow(X_npy[slice, :, :], cmap='gray', aspect='equal', alpha=(1-alpha))
                            if mode == 3:
                                plt.imshow(X_cam, cmap='jet', aspect='equal', alpha=alpha)
                            plt.title('True_label: ' + true_label)
                            plt.xlabel('Pred_label: ' + pred_label)
                            plt.ylabel(idx)
                    fig_name = os.path.join(curr_folder, '_'.join(file_name[:-1]) + '.png')
                    # plt.show()
                    plt.tight_layout()
                    plt.savefig(fig_name)
                    plt.close('all')
                    print(fig_name)

def create_all_in_one_excel(experiment_folder_name, imgs_folder_name):
    base_path = r'/home/jovyan/shared/InteliRad-gasper'
    experiment_path = os.path.join(base_path, experiment_folder_name)
    imgs_folder_path = os.path.join(base_path, imgs_folder_name)
    features_and_references_file_name = 'Data/features_and_references_dataframe_1866'
    features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))
    os.chdir(experiment_path)
    num_of_slices = 3
    res_folders = [filename for filename in os.listdir('.') if filename.startswith("RES")]
    excel_templata_df = pd.read_pickle(os.path.join(base_path, 'excel_templata'))
    final_df = pd.DataFrame()

    for folder in res_folders:
        excel_templata_df_copy = excel_templata_df.copy()
        curr_folder = os.path.join(experiment_path, folder)
        os.chdir(curr_folder)
        excel_name = "Results.xlsx"
        df_excel = pd.read_excel(excel_name)
        res_data = df_excel.iloc[:11, :]
        indeksi = res_data.iloc[:, 0].to_list()
        res_data = res_data.drop(['Input'], axis=1)
        res_data.index = indeksi
        meta_data_list = [''] + df_excel.iloc[11:, 0].to_list()

        for col in res_data.columns:
            for idx in res_data.index:
                excel_templata_df_copy.loc[idx, col] = res_data.loc[idx, col].round(3)
            excel_templata_df_copy.loc['AVG', col] = res_data.loc[indeksi[:-2], col].mean().round(3)
            excel_templata_df_copy.loc['AVG_vsi eksperimenti', col] = res_data.loc[indeksi, col].mean().round(3)
        st_stolpcev = excel_templata_df_copy.shape[1]
        for i in meta_data_list[::-1]:
            new_row = ['']*st_stolpcev
            new_row[1] = i
            excel_templata_df_copy = Insert_row(0, excel_templata_df_copy, new_row)

        final_df = final_df.append(excel_templata_df_copy)

    os.chdir(experiment_path)
    file_name = 'results_merged' + '.xlsx'
    writer = pd.ExcelWriter(file_name)
    final_df.to_excel(writer, sheet_name='Sheet1', header=True, index=False)
    writer.save()

def create_cam_random_images(experiment_folder_name, imgs_folder_name):
    base_path = r'/home/jovyan/shared/InteliRad-gasper'
    experiment_path = os.path.join(base_path, experiment_folder_name)
    imgs_folder_path = os.path.join(base_path, imgs_folder_name)
    features_and_references_file_name = 'Data/features_and_references_dataframe_1866'
    features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))
    os.chdir(experiment_path)
    num_of_slices = 3
    res_folders = [filename for filename in os.listdir('.') if filename.startswith("RES")]
    device_serial_numbers = ['11018', '45321', '70982', '21911', '000000SI4024MR02', '22002', '41597', '141797',
                             '35198', 'B', 'C']
    for folder in res_folders:
        folder_path = os.path.join(experiment_path, folder)
        mode = int(folder.split('_')[-2])
        if mode == 3:
            for ser_num in device_serial_numbers:
                dataset_folder_path = os.path.join(folder_path, ser_num)
                os.chdir(dataset_folder_path)
                model_path = os.path.join(dataset_folder_path, ser_num + '_VGG16.h5')
                final_img_name = 'CAM_random_30'
                rows = 5
                cols = 6
                alpha = 1 - 0.65

                os.chdir(imgs_folder_path)
                tmp_df = pd.read_pickle('test_' + ser_num + '_df')
                X_npy = np.load('test_' + ser_num + '.npy')
                izbrani = random.sample(set(tmp_df.index.to_list()), 30)
                idxs = []
                for i in izbrani:
                    idxs.extend(np.arange(tmp_df.shape[0])[tmp_df.index.isin([i])])
                nepodvojeni = [not (i) for i in tmp_df.loc[izbrani].index.duplicated()]
                idxs = np.array(idxs)[nepodvojeni]
                true_labels = features_and_references_dataframe.loc[izbrani]['sequence_contrast']

                loaded_model = keras.models.load_model(model_path)

                fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
                for i, (idx, seq, izb_name) in enumerate(zip(idxs, true_labels, izbrani)):
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

                plt.savefig(os.path.join(dataset_folder_path, final_img_name))
                plt.close('all')

def create_cam(loaded_model, X_npy, already_3channel = False):
    gap_weights = loaded_model.layers[-1].get_weights()[0]
    cam_model = Model(inputs=loaded_model.input,
                      outputs=(loaded_model.layers[-3].output, loaded_model.layers[-1].output))
    X_out = np.zeros_like(X_npy)
    y_out = []
    if loaded_model.output_shape[1] == 4:
        modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER']
    elif loaded_model.output_shape[1] == 5:
        modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER', 'T1W_CONTRAST']
    elif loaded_model.output_shape[1] == 3:
        modalitete = ['CN', 'MCI', 'AD']

    if already_3channel:
        if X_npy.ndim == 3:
            features, results = cam_model.predict(X_npy[np.newaxis, :])
        elif X_npy.ndim == 4:
            features, results = cam_model.predict(X_npy)
        pred = np.argmax(results, axis=1)
        cam_weights = gap_weights[:, pred]
        cam_output = np.dot(features, cam_weights)[0]
        y_out.append(np.asarray(modalitete)[pred])

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

def save_model_structure(model, location, name):
    plot_model(model, to_file=os.path.join(location, name + '.png'))
    print(os.path.join(location, name + '.png'))

def save_model_summary(model, location, name):
    with open(os.path.join(location, name + '.txt'),'w+') as f:
        with redirect_stdout(f):
            model.summary()
    print(os.path.join(location, name + '.txt'))

def save_model_info(experiment_folder_name):
    modes = [1, 2, 3]
    for mode in modes:
        model = initialize_VGG16(mode=mode, num_of_outputs=4, image_size=128)
        save_model_structure(model, experiment_folder_name, 'skica_modela_mode=' + str(mode))
        save_model_summary(model, experiment_folder_name, 'model_summary_mode=' + str(mode))

def save_dataset(npy_X, df_y, description, final_folder_path):
    os.chdir(final_folder_path)
    # npy_X, df_y = mix_order(npy_X, df_y)
    np.save(description + '.npy', npy_X)
    # with open(description + '.txt', 'wb') as fp:
    #     pickle.dump(npy_y, fp)
    df_y.to_pickle(description + '_df')
    print('SAVED!')

def normalize_CAM(X_cam):
    return X_cam/np.sum(np.sum(X_cam, axis=1), axis=1)[:, np.newaxis, np.newaxis]

def each_set_preprocess(list_of_names, df, npy, mode, modalitete):
        X, y_df = prepare_3_channel_np_arrays(list_of_names,
                                                       df,
                                                       npy,
                                                       num_of_sices=3,
                                                       mode=mode)
        if mode == 2:  # three same images in three channels
            X = [X[i] for i in range(3)]

        y_dummies = to_dummies(y_df, modalitete)
        return X, y_df, y_dummies

def prepare_X_y_adni(X_in, y_in_df, indices, mode, num_of_imgs=3, image_size = 128, ref_col_name = 'disease_status'):
    y_out_df = pd.DataFrame()
    if mode == 1 or mode == 3:
        X_out = np.zeros((len(indices), image_size, image_size, num_of_imgs))
        for enum, idx in enumerate(indices):
            slices = np.arange(y_in_df.shape[0])[y_in_df['Image Data ID'] == idx]
            # print(slices)
            X_out[enum] = np.moveaxis(standardize(X_in[slices]), 0, -1)
            y_out_df = y_out_df.append(y_in_df.iloc[slices[0]])
    elif mode == 2:
        X_out_i = np.zeros((3, len(indices), image_size, image_size, num_of_imgs))
        list_tmp = y_in_df.index.to_list()
        for iter, idx in enumerate(indices):
            slices = np.arange(y_in_df.shape[0])[y_in_df['Image Data ID'] == idx]
            y_out_df = y_out_df.append(y_in_df.iloc[slices[0]])
            for pos, s in enumerate(slices):
                X_out_i[pos, iter, :, :, :] = np.stack((standardize(X_in[s]),) * 3, axis=-1)
        X_out = [X_out_i[0], X_out_i[1], X_out_i[2]]
    return X_out, y_out_df

if __name__ == "__main__":
    print('Nič')
