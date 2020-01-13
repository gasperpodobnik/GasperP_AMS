import modality_experiments
import datetime
import os
import funkcije

# import tensorflow as tf
# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

danes = str(datetime.date.today().day) + '_' + str(datetime.date.today().month) + '_' + str(datetime.date.today().year)
cover_results_folder = 'eksperimenti_3_1_2020_vsi'
imgs_folder_name = 'PREPARED_IMAGES_slice_choice_COR_resolution_fixed_levo_desno_fixed'
# contrast_case = 1

# for contrast_case, i in zip([1, 0, 1], [2, 3, 3]):
#     vgg = modality_experiments.modality_experiment(contrast_case=contrast_case)
#     vgg.NUM_EPOCHS_vgg16 = 20
#     vgg.npy_mode = i
#     vgg.learning_rate = 1e-4
#     exp_name = 'RES_' + danes + '_EPOCHS_' + str(vgg.NUM_EPOCHS_vgg16) + '_LR_' + str(vgg.learning_rate) + '_SLICE_MODE_' + str(vgg.npy_mode) + '_CONTRAST=' + str(contrast_case)
#     vgg.excel_folder_name = os.path.join(cover_results_folder, exp_name)
#     vgg.api()
#     del vgg


# funkcije.create_all_in_one_excel(experiment_folder_name=cover_results_folder, imgs_folder_name=imgs_folder_name)
funkcije.izris_nap_klasif(experiment_folder_name=cover_results_folder, imgs_folder_name=imgs_folder_name)
# funkcije.create_cam_random_images(experiment_folder_name=cover_results_folder, imgs_folder_name=imgs_folder_name)
# funkcije.save_model_info(experiment_folder_name=cover_results_folder)


#
#
# for i in range(2):
#     vgg = modality_experiments.modality_experiment(contrast_case=contrast_case)
#     vgg.NUM_EPOCHS_vgg16 = 50
#     vgg.npy_mode = i+1
#     vgg.learning_rate = 1e-3
#     vgg.excel_folder_name = 'RES_' + danes + '_EPOCHS_' + str(vgg.NUM_EPOCHS_vgg16) + '_LR_' + str(vgg.learning_rate) + '_SLICE_MODE_' + str(vgg.npy_mode) + '_CONTRAST=' + str(contrast_case)
#     vgg.api()
#     del vgg
#
#
# for i in range(2):
#     vgg = modality_experiments.modality_experiment(contrast_case=contrast_case)
#     vgg.NUM_EPOCHS_vgg16 = 25
#     vgg.npy_mode = i+1
#     vgg.learning_rate = 1e-4
#     vgg.excel_folder_name = 'RES_' + danes + '_EPOCHS_' + str(vgg.NUM_EPOCHS_vgg16) + '_LR_' + str(vgg.learning_rate) + '_SLICE_MODE_' + str(vgg.npy_mode) + '_CONTRAST=' + str(contrast_case)
#     vgg.api()
#     del vgg
#
# for i in range(2):
#     vgg = modality_experiments.modality_experiment(contrast_case=contrast_case)
#     vgg.NUM_EPOCHS_vgg16 = 10
#     vgg.npy_mode = i+1
#     vgg.learning_rate = 1e-4
#     vgg.excel_folder_name = 'RES_' + danes + '_EPOCHS_' + str(vgg.NUM_EPOCHS_vgg16) + '_LR_' + str(vgg.learning_rate) + '_SLICE_MODE_' + str(vgg.npy_mode) + '_CONTRAST=' + str(contrast_case)
#     vgg.api()
#     del vgg