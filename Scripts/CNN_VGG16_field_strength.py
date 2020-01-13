from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
from keras import models
from keras import layers
from keras import optimizers
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import keras
import pickle
from funkcije import writeResToExcel, saveROC


base_path = r'/home/jovyan/shared/InteliRad-gasper'
os.chdir(os.path.join(base_path, 'PREPARED_IMAGES'))
EXP_NP_FILE_NAME = ['4slices_3dim_Gauss_CC_359']
with open(EXP_NP_FILE_NAME[0] + '_df', 'rb') as fp:
    feat_df = pickle.load(fp)
# feat_df.index[feat_df['vendor'] == 'ge'].tolist()

features_names = ['vendor', 'field_strength', 'age', 'sex']
features_unique = []
for f in features_names:
    features_unique.append(feat_df[f].nunique())


for i_NP_FILE_NAME in EXP_NP_FILE_NAME:
    for idx_f, feature in enumerate(features_names):

        SEED = 49
        NP_FILE_NAME = i_NP_FILE_NAME
        NUM_EPOCHS = 50
        LR = 1e-4

        ####### PODATKI ZA VPIS V EXCEL TABELO ###################
        # OPIS SLIK e.g. 4 rezine na sliko, samo sagitalni prerezi
        opis_slik = NP_FILE_NAME

        # IZBIRA REZIN e.g. Gauss, mean=dim/2, std=dim/4
        izbira_rezin = feature

        # MODALITETE ['T1W','T2W','FLAIR','OTHER'], 'T1W_CONTRAST'
        # modalitete = i_modalitete
        # if all((m == 'T1W' or m == 'T1W_CONTRAST') for m in modalitete):
        #     DELETE_IF_NOT_IN_MODALITETE = 1
        # else:
        #     DELETE_IF_NOT_IN_MODALITETE = 0

        #  NASTAVITVE e.g. VGG16, 50 epochs, RMSprop(lr=1e-4)
        nastavitve_CNN = 'VGG16, {} epochs, RMSprop(lr={}), seed={}'.format(NUM_EPOCHS, LR, SEED)
        excel_name = 'Rezultati_CNN_VGG16_CC_359_round2'

        save_results = True


        base_path = r'/home/jovyan/shared/InteliRad-gasper'


        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     UPLOAD BEFORE RUNNING      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        image_size = 128

        vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

        # Freeze the layers except the last 4 layers
        for layer in vgg_conv.layers[:-4]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in vgg_conv.layers:
            print(layer, layer.trainable)

        vgg_conv.summary()

        # Create the model
        model = models.Sequential()

        # Add the vgg convolutional base model
        model.add(vgg_conv)

        # Add new layers
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(features_unique[idx_f], activation='softmax'))

        # Show a summary of the model. Check the number of trainable parameters
        model.summary()

        os.chdir(os.path.join(base_path, 'PREPARED_IMAGES'))
        X = np.load(NP_FILE_NAME + '.npy')
        X = np.stack((X,)*3, axis=-1)

        unq = feat_df[feature].unique()
        y = feat_df[feature].to_list()

        y = pd.DataFrame({'ref': y})

        y = y.astype('category', categories=unq)
        y = pd.get_dummies(y)
        y_np = y.values

        X_train, X_test, y_train, y_test = train_test_split(X,y_np, random_state=SEED)


        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=LR),
                      metrics=['acc'])

        model.fit(X_train, y_train,
                  epochs=NUM_EPOCHS,
                  validation_data=(X_test, y_test))

        score_train = model.evaluate(X_train, y_train, verbose=0)
        score_test = model.evaluate(X_test, y_test, verbose=0)

        y_pred = model.predict(X_test)


        roc_micro, roc_macro = saveROC(y_test, y_pred, save_results)

        if save_results:
            results = {'Input': opis_slik,
                 'Input_opis': izbira_rezin,
                 'Modalitete': ', '.join(' '),
                 'Nastavitve': nastavitve_CNN,
                 'Train_acc': score_train[1],
                 'Test_acc': score_test[1],
                 'Train_loss': score_train[0],
                 'Test_loss': score_test[0],
                 'ROC_micro': roc_micro,
                 'ROC_macro': roc_macro}

            writeResToExcel(excel_name, results)


        print('Test loss:', score_test[0])
        print('Test accuracy:', score_test[1])

# tensorflow version 1.4.0 and keras version 2.0.8.
# CUDA 8.0
# cudnn 6.0