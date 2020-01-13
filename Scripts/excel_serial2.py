import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
import numpy as np
import funkcije


image_size = 128
num_of_slices = 4  # set number of slices per plane (sagittal, axial, coronal) per each image; mora biti sodo
num_of_dim = 3

base_path = r'/home/jovyan/shared/InteliRad-gasper'
img_base_path = os.path.join(base_path, 'images')
npy_base_path = os.path.join(base_path, 'PREPARED_IMAGES')
results_folder = '/home/jovyan/shared/InteliRad-gasper/REZULTATI/'
os.chdir(npy_base_path)

features_and_references_file_name = 'features_and_references_dataframe_1866'
features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))

datasets = [['11018'], ['45321'], ['70982'], ['21911'], ['000000SI4024MR02'], ['22002'], ['41597'], ['141797'], ['35198'], ['22772'], ['000000SI4025MR01'], ['000000SI4024MR01'], ['32283'], ['67063'], ['49134'], ['17260'], ['35033'], ['49143'], ['70826'], ['35028'], ['000000007579533T']]

vse_modalitete = ['T1W', 'T1W_CONTRAST', 'T2W', 'FLAIR', 'OTHER', 'SPINE_OTHER', 'SPINE_T2W', 'SPINE_T1W', 'SPINE_FLAIR']

for experiment_num, dataset in enumerate(datasets):
    os.chdir(base_path)
    dataset = dataset[0]

    file_name = 'Dataset_num_of_imgs_contrast_all_modal' + '.xlsx'

    if not (os.path.isfile(file_name)):
        writer = pd.ExcelWriter(file_name)  # , engine='xlsxwriter')
        header = ['Serial number', 'Number of imgs']
        header.extend(vse_modalitete)
        pd.DataFrame([header]).to_excel(writer, sheet_name='Sheet1', header=False, index=False)
        writer.save()

    content = pd.read_excel(file_name)
    col_names = content.columns.values
    cont_np = content.values
    cont_np = np.concatenate(([col_names], cont_np))

    # tmp_df = features_and_references_dataframe[(features_and_references_dataframe.sequence == 'T1W') &
    #                                            (features_and_references_dataframe.pravi3D == 1) &
    #                                            (features_and_references_dataframe.DeviceSerialNumber == dataset)]
    tmp_df = features_and_references_dataframe[(features_and_references_dataframe.pravi3D == 1) &
                                               (features_and_references_dataframe.DeviceSerialNumber == dataset)]

    number_of_imgs = tmp_df.shape[0]
    stevec_modalitet = np.zeros((len(vse_modalitete)))
    # for enum, modaliteta in enumerate(vse_modalitete):
    #     stevec_modalitet[enum] = int(np.array((tmp_df['hasContrast (0/1)'] == str(enum)).to_list()).sum())
    stevec_modalitet[0] = int(np.array(((tmp_df.sequence == 'T1W') & (tmp_df['hasContrast (0/1)'] == str(0))).to_list()).sum())
    stevec_modalitet[1] = int(np.array(((tmp_df.sequence == 'T1W') & (tmp_df['hasContrast (0/1)'] == str(1))).to_list()).sum())
    for i in np.arange(2, len(vse_modalitete)):
        stevec_modalitet[i] = int(np.array((tmp_df.sequence == vse_modalitete[i]).to_list()).sum())


    new_np = [dataset, str(number_of_imgs)]
    new_np.extend([str(i) for i in stevec_modalitet])
    np_to_write = np.concatenate((cont_np, np.asarray(new_np).reshape((1,len(new_np)))))
    to_write = pd.DataFrame(np_to_write)
    writer = pd.ExcelWriter(file_name)
    to_write.to_excel(writer, index=False, header=None)
    writer.save()



