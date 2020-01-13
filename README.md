# GasperP_AMS: Globoke nevronske mreže za zgodnje zaznavanje nevroloških bolezni iz MR slik
Koda in modeli za seminar pri predmetu Analiza medicinskih slik

## Requirements
V datoteki req.txt so zapisane knjižnice, za poganjanje eksperimentov. Za upoabo model vseh teh knjižnic ni potrebno imeti naloženih. 
## Models
Ker datoteke z utežmi modelov presegajo omejitev 25 MB, prilagam povezavo do moje Google Drive računa: https://drive.google.com/drive/folders/1MfSbVUqwaNObokJX3iQaMFXZqgbjVDin?usp=sharing

| problem    | model name                                     | accuracy on test set |   |
|------------|------------------------------------------------|----------------------|---|
| Modelities | adni_ss_158_159_160.h5                         |                      |   |
| Modelities | adni_ss_179_180_181.h5                         |                      |   |
| Modelities | adni_v2_179_180_181.h5                         |                      |   |
| Alzheimer  | modalities_model1_T1W_T2W_FLAIR_OTHER_VGG16.h5 |                      |   |
| Alzheimer  | modalities_model2_T1W_T2W_FLAIR_OTHER_VGG16.h5 |                      |   |
| Alzheimer  | modalities_model3_T1W_T2W_FLAIR_OTHER_VGG16.h5 |                      |   |

## Data
Iz enakih razlogov kot zgoraj prilagam povezavo do moje Google Drive računa, kjer so na voljo podatki s katerimi se lahko poganja eksperimente: https://drive.google.com/drive/folders/1zNRPShB3ZlRaUb109Xt4feCabqcX58Jq?usp=sharing

| problem    | model name                                     | opis                                     |   |
|------------|------------------------------------------------|------------------------------------------|---|
| Alzheimer  | adni_images_ss_179_180_181.npy                 |  ndarray s slikami                       |   |
| Alzheimer  | adni_true_labels_ss_179_180_181_df             |  referencni podatki za zgornji ndarray   |   |
| Modelities | modalities_images_dataset_B.npy                |  ndarray s slikami                       |   |
| Modelities | modalities_true_labels_dataset_B_df            |  referencni podatki za zgornji ndarray   |   |

## Examples
Skripta adni_example.py prikaže način uporabe modela za razvrščanje v [CN, MCI, AD] na podatkih iz mape Data
Skripta modalities_example.py prikaže način uporabe modela za razvrščanje v [T1w, T2w, FLAIR, OTHER] na podatkih iz mape Data
