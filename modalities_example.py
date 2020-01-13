import sys
sys.path.insert(0, './Scripts')
import funkcije, funkcije_adni
import numpy as np
import pandas as pd
import os, sys
import keras

## example that uses model for classifying MRI images into modalities group [T1w, T2w, FLAIR, OTHER]
# choose which dataset to you for prediction
dataset_name = 'modalities_images_dataset_B'
model_name = 'modalities_model2_T1W_T2W_FLAIR_OTHER_VGG16.h5'
mode = int(model_name.split('_')[1][-1])
# path to save graphs
save_path = os.path.join(os.getcwd(), 'MODALITIES_results')
modalitete = ['T1W', 'T2W', 'FLAIR', 'OTHER']

# Import data
X_all = np.load(os.path.join('Data', dataset_name + '.npy'))
y_all_df = pd.read_pickle(os.path.join('Data', dataset_name + '_df'))

test_npy, test_df = funkcije.load_datasets('Data',
                                           dataset_name,
                                           modalitete)

# Import model
model_path = os.path.join('Models', model_name)
model = keras.models.load_model(model_path)

X_test, y_test_df, y_test_dummies = funkcije.each_set_preprocess(list(set(test_df.index.to_list())),
                                                                 test_df,
                                                                 test_npy,
                                                                 mode = mode)

y_test_predicted = model.predict(X_test)
score = model.evaluate(X_test, y_test_dummies, verbose=0)

print('SCORE: ' + str(score))
funkcije.saveROC(y_test_dummies,
                 y_test_predicted,
                 True,
                 save_path,
                 dataset_name)
funkcije.plot_confusion_matrix(y_test_dummies,
                               y_test_predicted,
                               modalitete,
                               save_path,
                               title=str(dataset_name) + '_test_cf')

