import pandas as pd
import os, sys
import numpy as np

base_path = r'/home/jovyan/shared/InteliRad-gasper'
res_path = os.path.join(base_path, 'REZULTATI2')
excel_file_name = 'archive_analysis' + '.xlsx'
excel_file_path = os.path.join(base_path, excel_file_name)
archive_df = pd.read_pickle(os.path.join(base_path,'archive_df'))

if os.path.isfile(excel_file_path):
    os.remove(excel_file_name)

# CASE: T1W, T2W, FLAIR, OTHER
modalities = ['T1W', 'T2W', 'FLAIR', 'OTHER']
for modal in modalities:
    base_modalities = ['T1W', 'T2W', 'FLAIR', 'OTHER']
    if 'T1W' in modal:
        true_modal = ['T1W_NO_CONTRAST', 'T1W_CONTRAST']
    else:
        true_modal = [modal]
    df_tmp = archive_df.loc[archive_df['TrueSequence'].isin(true_modal)]

    base_modalities.remove(modal)
    df_tmp.loc[:,'Total'] = df_tmp[base_modalities].sum(axis=1)
    df_tmp = df_tmp.sort_values(by=['Total'], ascending=False)
    if not(os.path.isfile(excel_file_path)):
        writer = pd.ExcelWriter(excel_file_path)#, engine='xlsxwriter')
        df_tmp.iloc[0:10,:].to_excel(writer, index=True, header=True)
        writer.save()
    else:
        content = pd.read_excel(excel_file_path, header=0, index_col=0)
        writer = pd.ExcelWriter(excel_file_path)
        to_write = pd.concat((content,df_tmp.iloc[0:10,:]), axis=0, sort=False)
        to_write.to_excel(writer, index=True, header=True)
        writer.save()

# CASE: T1W_NO_CONTRAST, T1W_CONTRAST
modalities = ['T1W_NO_CONTRAST', 'T1W_CONTRAST']
for modal in modalities:
    base_modalities = ['T1W_NO_CONTRAST', 'T1W_CONTRAST']
    true_modal = [modal]
    df_tmp = archive_df.loc[archive_df['TrueSequence'].isin(true_modal)]
    base_modalities.remove(modal)
    df_tmp.loc[:,'Total'] = df_tmp[base_modalities].sum(axis=1)
    df_tmp = df_tmp.sort_values(by=['Total'], ascending=False)
    if not(os.path.isfile(excel_file_path)):
        writer = pd.ExcelWriter(excel_file_path)#, engine='xlsxwriter')
        df_tmp.iloc[0:10,:].to_excel(writer, index=True, header=True)
        writer.save()
    else:
        content = pd.read_excel(excel_file_path, header=0, index_col=0)
        writer = pd.ExcelWriter(excel_file_path)
        to_write = pd.concat((content,df_tmp.iloc[0:10,:]), axis=0, sort=False)
        to_write.to_excel(writer, index=True, header=True)
        writer.save()

