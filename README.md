# GasperP_AMS: Deep neural networks for early diagnosis of neurological disorders from MR images
Code and trained models that were developed as a project under subject Analysis of medical images

### The purpose of this project is to offer trained model:
### 1. for MRI modality classification [T1w, T2w, FLAIR, OTHER]; problem = Modalities
### 2. for classification of MRI images from ADNI database in three groups of people: cognitively normal (CN), mild cognitive impairment (MCI) and Alzheimer's disease (AD); problem = Alzheimer

## Requirements
Required libraries for running scripts is summarized in file req.txt
Note that not all libraries are needed if you intend to just use this models for prediction

## Models
Due to the fact that files with model weights exceed 25 MB, I attach these files in the following link to my Google Drive: https://drive.google.com/drive/folders/1MfSbVUqwaNObokJX3iQaMFXZqgbjVDin?usp=sharing

#### For all models three (preferably consecutive) slices with shape 128x128 are needed: [1,2,3]

| problem     | model name                                     | input shape          | slices  |
|-------------|------------------------------------------------|----------------------|---|
| Modalities  | modalities_model1_T1W_T2W_FLAIR_OTHER_VGG16.h5 | (None, 128, 128, 3) | [1,2,3]  |
| Modalities  | modalities_model2_T1W_T2W_FLAIR_OTHER_VGG16.h5 | [(None, 128, 128, 3), (None, 128, 128, 3), (None, 128, 128, 3)]   | [[1,1,1], [2,2,2], [3,3,3]]  |
| Modalities  | modalities_model3_T1W_T2W_FLAIR_OTHER_VGG16.h5 | (None, 128, 128, 3)  | [1,2,3]  |
| Alzheimer   | adni_ss_158_159_160.h5                          |  (None, 128, 128, 3) | [1,2,3]  |
| Alzheimer   | adni_ss_179_180_181.h5                          |  (None, 128, 128, 3) | [1,2,3]  |
| Alzheimer   | adni_v2_179_180_181.h5                          |  (None, 128, 128, 3) | [1,2,3]  |


## Data
Due to same reasons as outlined above, I attach example data in my Google Drive folder: https://drive.google.com/drive/folders/1zNRPShB3ZlRaUb109Xt4feCabqcX58Jq?usp=sharing

| problem    | ndarray name                                   | true labels dataframe                    |
|------------|------------------------------------------------|------------------------------------------|
| Modalities | modalities_images_dataset_B.npy                |  modalities_true_labels_dataset_B_df     |
| Alzheimer  | adni_images_ss_179_180_181.npy                 |  adni_true_labels_ss_179_180_181_df      | 

## Examples
Python script adni_example.py shows how to use model for classification into the following classes [CN, MCI, AD] based on example data in folder Data

Python script modalities_example.py shows how to use model for MRI modalities classification into the following classes [T1w, T2w, FLAIR, OTHER] based on example data in folder Data

## Literature
Please refer to two PDF reports for more details
