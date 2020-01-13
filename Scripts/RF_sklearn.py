import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statistics
from sklearn.model_selection import cross_val_score
import os
import pickle
from funkcije import writeResToExcel, saveROC



NP_FILE_NAME = '4slices_3dim_Gauss'
N_ESTIMATORS = 100
MIN_SMPL_LEAF = 10
CRITERION = 'entropy'

####### PODATKI ZA VPIS V EXCEL TABELO ###################
# OPIS SLIK e.g. 4 rezine na sliko, samo sagitalni prerezi
opis_slik = NP_FILE_NAME

# IZBIRA REZIN e.g. Gauss, mean=dim/2, std=dim/4
parametri = ['RepetitionTime', 'EchoTime', 'InversionTime', 'EchoTrainLength', 'PercentSampling', 'PercentPhaseFieldofView', 'PixelBandwidth']
izbira_rezin = 'DICOM parametri: ' + ', '.join(parametri) # + '_samo T1W in T1W_C'

# MODALITETE ['T1W','T2W','FLAIR','OTHER'], 'T1W_CONTRAST'
modalitete = ['T1W', 'T1W_CONTRAST']
DELETE_IF_NOT_IN_MODALITETE = 1


#  UPORABLJENI PARAMETRI e.g. ['RepetitionTime', 'EchoTime', 'InversionTime']
nastavitve_RF = 'RF_sklearn, {} estimators, min samples leaf: {}, criterion: {})'.format(N_ESTIMATORS, MIN_SMPL_LEAF, CRITERION)

excel_name = 'Rezultati_CNN_VGG16_dicom'

save_results = True


base_path = r'/home/jovyan/shared/InteliRad-gasper'


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     UPLOAD BEFORE RUNNING      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

os.chdir(os.path.join(base_path, 'PREPARED_IMAGES'))
ref_data = pd.read_pickle(NP_FILE_NAME + '_df')

model_2D_3D_filename = '2D_3D_model.sav'
model_2D_3D = pickle.load(open(os.path.join(base_path,model_2D_3D_filename), 'rb'))

X = ref_data[parametri]
X = X.replace('N-A',-1)
X = X.astype(float)

X_prep = np.zeros_like(X.values)

y_modality = ref_data['ref_sequence'].values
y_contrast = ref_data['ref_contrast'].values

y = []
idx_stay = 0
for i, mdl in enumerate(y_modality):
    if mdl == 'T1W' and y_contrast[i] == '1' and 'T1W_CONTRAST' in modalitete:
        y.append('T1W_CONTRAST')
        if DELETE_IF_NOT_IN_MODALITETE:
            X_prep[idx_stay,:] = X.iloc[i].values
            idx_stay += 1
    elif mdl not in modalitete:
        if DELETE_IF_NOT_IN_MODALITETE:
            pass
        else:
            y.append('OTHER')
    else:
        y.append(mdl)
        if DELETE_IF_NOT_IN_MODALITETE:
            X_prep[idx_stay,:] = X.iloc[i].values
            idx_stay += 1

y = pd.DataFrame({'ref': y})
if DELETE_IF_NOT_IN_MODALITETE:
    X_prep = X_prep[:idx_stay]
else:
    X_prep = X

y = y.astype('category', categories=modalitete)
y = pd.get_dummies(y)
y_np = y.values

X_train, X_test, y_train, y_test = train_test_split(X_prep,y_np)

clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SMPL_LEAF, criterion=CRITERION)
clf.fit(X_train, y_train)

score_train = clf.score(X_train, y_train)
score_test = clf.score(X_test, y_test)

y_pred = clf.predict(X_test)

roc_micro, roc_macro = saveROC(y_test, y_pred, save_results)


if save_results:
    results = {'Input': opis_slik,
         'Input_opis': izbira_rezin,
         'Modalitete': ', '.join(modalitete),
         'Nastavitve': nastavitve_RF,
         'Train_acc': score_train,
         'Test_acc': score_test,
         'Train_loss': '/',
         'Test_loss': '/',
         'ROC_micro': roc_micro,
         'ROC_macro': roc_macro}

    writeResToExcel(excel_name, results)

print('Test loss:', score_test)
print('Test accuracy:', score_test)
