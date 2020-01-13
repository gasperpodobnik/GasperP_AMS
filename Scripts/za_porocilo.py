import numpy as np
from os.path import exists, join
import pandas as pd
import glob
import SimpleITK as sitk
import os, sys
import pickle
import funkcije
from funkcije import resasample_to_size
from slice_choice_control import get_slices
import matplotlib.pyplot as plt
import matplotlib

image_size = 128
slice_plane = 'cor'
choose_slices = [-15, 0, 15]
final_folder_name = 'PREPARED_IMAGES_slice_choice_COR_resolution_fixed_levo_desno_fixed'

# File locations
base_path = r'/home/jovyan/shared/InteliRad-gasper'
img_base_path = join(base_path, 'images')
final_folder_path = os.path.join(base_path, final_folder_name)
os.chdir(final_folder_path)

# Import dataframe
features_and_references_file_name = 'features_and_references_dataframe_1866'
features_and_references_dataframe = pd.read_pickle(os.path.join(base_path, features_and_references_file_name))

# Get all DeviceSerialNumbers
device_serial_numbers = [['11018'],['45321'], ['70982'], ['21911'], ['000000SI4024MR02'], ['22002'], ['41597'], ['141797'], ['35198'], ['22772', '000000SI4025MR01', '000000SI4024MR01','32283', '67063', '49134', '17260', '35033', '49143', '70826', '35028', '000000007579533T' ]]

npy = np.load(os.path.join(final_folder_path, 'train_B.npy'))
df = pd.read_pickle(os.path.join(final_folder_path, 'train_B_df'))

other1_path = 'MR103_MR VEN_BOLD_SWI_HEMO'
other2_path = 'MR99_MR DWI_og 2 NSAsat'
other3_path = 'MR77_MR t1_tse_tra_FS KS_CH'
other4_path = 'MR160_MR Pha_Images'

t1w_no1_path = 'MR01_MR MPR AX-2'
t1w_no2_path = 'MR06_MR sT1W_3D'
t1w_no3_path = 'MR100_MR T1W_SE'

t1w_c1_path = 'MR01_MR eT1W_SE KS CLEAR'
t1w_c2_path = 'MR108_MR eT1W_SE KS CLEAR'
t1w_c3_path = 'MR100_MR T1W_SE KS'

flair1_path = 'MR01_MR eFLAIR 3D SENSE'
flair2_path = 'MR07_MR FLAIR 2nsa'
flair3_path = 'MR105_MR FLAIR 3D'

t2w1_path = 'MR01_MR eT2FILTER'
t2w2_path = 'MR103_MR T2W_TSE CLEAR'
t2w3_path = 'MR118_MR eT2FILTER'

t1w_no_ozn = 'T1W NO CONTRAST'
t1w_c_ozn = 'T1W CONTRAST'
t2w_ozn = 'T2W'
flair_ozn = 'FLAIR'
other_ozn = 'OTHER'

plt.close('all')

def get_idx(df, idx):
    return np.arange(df.shape[0])[df.index.isin([idx])]
pathi = [t1w_no1_path, t1w_c1_path, t2w1_path, flair1_path, other1_path, other2_path, other3_path, other4_path]
mod = [t1w_no_ozn, t1w_c_ozn, t2w_ozn, flair_ozn, other_ozn, other_ozn, other_ozn, other_ozn]
rows = 2; cols = 4
fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
matplotlib.rcParams.update({'font.size': 18})
for e, i in enumerate(pathi):
    ax_tmp = ax[e // 4, e % 4]
    idx = get_idx(df, i)[1]
    ax_tmp.imshow(npy[idx], cmap='gray')
    ax_tmp.set_title(str(e+1) + ') ' + mod[e])
    ax_tmp.axis('off')
plt.savefig(os.path.join(base_path, 'primerki'))