# GasperP_AMS: Deep neural networks for early diagnosis of neurological disorders from MR images
Code and trained models that were developed as a project under subject Analysis of medical images

### The purpose of this project is to offer trained model:
### 1. for MRI modality classification [T1w, T2w, FLAIR, OTHER]; problem = Modalities
### 2. for classification of MRI images from ADNI database in three groups of people: cognitively normal, mild cognitive impairment and Alzheimer's disease; problem = Alzheimer

## Requirements
Required libraries for running scripts is summarized in file req.txt
Note that not all libraries are needed if you intend to just use this models for prediction

## Models
Due to the fact that files with model weights exceed 25 MB, I attach these files in the following link to my Google Drive: https://drive.google.com/drive/folders/1MfSbVUqwaNObokJX3iQaMFXZqgbjVDin?usp=sharing

| problem     | model name                                     | input shape          |   |
|-------------|------------------------------------------------|----------------------|---|
| Alzheimer   | adni_ss_158_159_160.h5                          |  (None, 128, 128, 3) |   |
| Alzheimer   | adni_ss_179_180_181.h5                          |  (None, 128, 128, 3) |   |
| Alzheimer   | adni_v2_179_180_181.h5                          |  (None, 128, 128, 3) |   |
| Modalities  | modalities_model1_T1W_T2W_FLAIR_OTHER_VGG16.h5 | (None, 128, 128, 3) |   |
| Modalities  | modalities_model2_T1W_T2W_FLAIR_OTHER_VGG16.h5 | [(None, 128, 128, 3), (None, 128, 128, 3), (None, 128, 128, 3)]   |   |
| Modalities  | modalities_model3_T1W_T2W_FLAIR_OTHER_VGG16.h5 | (None, 128, 128, 3)  |   |

## Data
Due to same reasons as outlined above, I attach example data in my Google Drive folder: https://drive.google.com/drive/folders/1zNRPShB3ZlRaUb109Xt4feCabqcX58Jq?usp=sharing

| problem    | ndarray name                                   | true labels dataframe                    |   |
|------------|------------------------------------------------|------------------------------------------|---|
| Alzheimer  | adni_images_ss_179_180_181.npy                 |  adni_true_labels_ss_179_180_181_df      |   |
| Modelities | modalities_images_dataset_B.npy                |  modalities_true_labels_dataset_B_df     |   |

## Examples
Skripta adni_example.py prikaže način uporabe modela za razvrščanje v [CN, MCI, AD] na podatkih iz mape Data

Skripta modalities_example.py prikaže način uporabe modela za razvrščanje v [T1w, T2w, FLAIR, OTHER] na podatkih iz mape Data

