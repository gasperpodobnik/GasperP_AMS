import os, sys
import numpy as np
import pandas as pd
import glob
from PIL import Image

base_path = r'/home/jovyan/shared/InteliRad-gasper'
res_path = os.path.join(base_path, 'REZULTATI2')
os.chdir(res_path)
folders = os.listdir()

for folder in folders:
    os.chdir(os.path.join(res_path,folder))
    if 'T1W_and_T1W_CONTRAST' in folder:
        seznam = ['11018', '22002', '141797', '21911', '70982', 'zdruzeno1', 'zdruzeno2']
    else:
        seznam = ['11018', '45321', '70982', '21911', '000000SI4024MR02', '22002', '41597', '141797', '35198',
                   'DATASET_B', 'DATASET_C']

    for enum, i in enumerate(seznam):
        cf_img = np.array(Image.open(i + '_test_cf.png'))
        roc_img = np.array(Image.open(i + '_ROC.png'))

        if enum == 0:
            final_array = np.concatenate((cf_img, roc_img), axis=0)
        else:
            final_array = np.concatenate(
                (final_array, np.concatenate((cf_img, roc_img), axis=0)),
                axis=1)
    Image.fromarray(final_array).save('excel_img_res.png')
    # writer = pd.ExcelWriter('Results.xlsx')#, engine='xlsxwriter')
    # workbook  = writer.book
    # worksheet = writer.sheets['Sheet1']
    # worksheet.insert_image('B25', 'excel_img_res.png')
    # writer.save()