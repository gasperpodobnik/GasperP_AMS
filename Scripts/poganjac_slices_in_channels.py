import modality_experiments
import datetime
import os

danes = str(datetime.date.today().day) + '_' + str(datetime.date.today().month) + '_' + str(datetime.date.today().year)
cover_results_folder = 'eksperimenti_resolution_fixed_levo_desno_fixed'
imgs_folder_name = 'PREPARED_IMAGES_slice_choice_COR_resolution_fixed_levo_desno_fixed'

vgg = modality_experiments.modality_experiment(contrast_case=0)
vgg.NUM_EPOCHS_vgg16 = 30
vgg.learning_rate = 1e-4 #'RAdam' #1e-3
vgg.npy_mode = 2
exp_name = 'RES_' + danes + '_EPOCHS_' + str(vgg.NUM_EPOCHS_vgg16) + '_LR_' + str(vgg.learning_rate) + '_SLICE_MODE_' + str(vgg.npy_mode)
vgg.excel_folder_name = os.path.join(cover_results_folder, exp_name)
vgg.api()

del vgg

# vgg = modality_experiments.modality_experiment(contrast_case=1)
# vgg.NUM_EPOCHS_vgg16 = 30
# vgg.learning_rate = 1e-4 #'RAdam' #1e-3
# exp_name = 'RES_' + danes + '_EPOCHS_' + str(vgg.NUM_EPOCHS_vgg16) + '_LR_' + str(vgg.learning_rate) + '_SLICE_MODE_' + str(vgg.npy_mode)
# vgg.excel_folder_name = os.path.join(cover_results_folder, exp_name)
# vgg.api()
#
# del vgg

#
# vgg = modality_experiments.modality_experiment(contrast_case=0)
# vgg.NUM_EPOCHS_vgg16 = 50
# vgg.learning_rate = 1e-3
# exp_name = 'RES_' + danes + '_EPOCHS_' + str(vgg.NUM_EPOCHS_vgg16) + '_LR_' + str(vgg.learning_rate) + '_SLICE_MODE_' + str(vgg.npy_mode)
# vgg.excel_folder_name = os.path.join(cover_results_folder, exp_name)
# vgg.api()
#
# del vgg
#
# vgg = modality_experiments.modality_experiment(contrast_case=0)
# vgg.NUM_EPOCHS_vgg16 = 25
# vgg.learning_rate = 1e-4
# exp_name = 'RES_' + danes + '_EPOCHS_' + str(vgg.NUM_EPOCHS_vgg16) + '_LR_' + str(vgg.learning_rate) + '_SLICE_MODE_' + str(vgg.npy_mode)
# vgg.excel_folder_name = os.path.join(cover_results_folder, exp_name)
# vgg.api()
#
# del vgg
#
# vgg = modality_experiments.modality_experiment(contrast_case=0)
# vgg.NUM_EPOCHS_vgg16 = 50
# vgg.learning_rate = 1e-4
# exp_name = 'RES_' + danes + '_EPOCHS_' + str(vgg.NUM_EPOCHS_vgg16) + '_LR_' + str(vgg.learning_rate) + '_SLICE_MODE_' + str(vgg.npy_mode)
# vgg.excel_folder_name = os.path.join(cover_results_folder, exp_name)
# vgg.api()
#
# del vgg
#
# vgg = modality_experiments.modality_experiment(contrast_case=0)
# vgg.NUM_EPOCHS_vgg16 = 25
# vgg.learning_rate = 1e-5
# exp_name = 'RES_' + danes + '_EPOCHS_' + str(vgg.NUM_EPOCHS_vgg16) + '_LR_' + str(vgg.learning_rate) + '_SLICE_MODE_' + str(vgg.npy_mode)
# vgg.excel_folder_name = os.path.join(cover_results_folder, exp_name)
# vgg.api()
#
# del vgg
#
# vgg = modality_experiments.modality_experiment(contrast_case=0)
# vgg.NUM_EPOCHS_vgg16 = 25
# vgg.learning_rate = 1e-6
# exp_name = 'RES_' + danes + '_EPOCHS_' + str(vgg.NUM_EPOCHS_vgg16) + '_LR_' + str(vgg.learning_rate) + '_SLICE_MODE_' + str(vgg.npy_mode)
# vgg.excel_folder_name = os.path.join(cover_results_folder, exp_name)
# vgg.api()