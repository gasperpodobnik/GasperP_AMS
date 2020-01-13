import sys
sys.path.insert(0, './Scripts')
import funkcije
import numpy as np
import pandas as pd
import os, sys
import keras

## example that uses model for classifying ADNI images in [cognitively normal, mild cognitive impairment and Alzheimer's demetia] group
# choose which dataset to you for prediction
dataset_name = 'adni_ss_179_180_181'
model_name = 'adni_v2_179_180_181.h5'
# path to save graphs
save_path = os.path.join(os.getcwd(), 'ADNI_results')
funkcije.create_folder(save_path)

# Import data
X_all = np.load(os.path.join('Data', dataset_name + '.npy'))
y_all_df = pd.read_pickle(os.path.join('Data', dataset_name + '_df'))
classes = ['CN', 'MCI', 'AD']
ref_col_name = 'disease_status'

# Import model
model_path = os.path.join('Models', model_name)
model = keras.models.load_model(model_path)

uniq_indices = list(set(y_all_df['Image Data ID']))
X_test, y_test = funkcije.prepare_X_y_adni(X_all, y_all_df, uniq_indices, mode=3)
y_test_dummies = pd.get_dummies(pd.Categorical(y_test[ref_col_name], categories=[0, 1, 2], ordered=True)).values

y_pred_dummies = model.predict(X_test)
funkcije.plot_confusion_matrix(y_test_dummies, y_pred_dummies, classes, save_path, title=dataset_name+'_CF')
funkcije.saveROC(y_test_dummies, y_pred_dummies, True, save_path, dataset_name)
