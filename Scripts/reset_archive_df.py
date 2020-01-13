import os, sys
import numpy as np
import pandas as pd
import glob

base_path = r'/home/jovyan/shared/InteliRad-gasper'
archive_df = pd.read_pickle(os.path.join(base_path,'archive_df'))
archive_df[archive_df.columns[:-1]] = 0
archive_df[archive_df.columns[:-1]].astype(np.int)
archive_df.to_pickle(os.path.join(base_path, 'archive_df'))