import os

base_path = r'/home/jovyan/shared/InteliRad-gasper'
npy_base_path = os.path.join(base_path, 'PREPARED_IMAGES_CONTRAST')
os.chdir(npy_base_path)
ll = os.listdir()
for file in ll:
    a = file.split('_')
    if a[1] == 'B2':
        a.remove(a[1])
        os.rename(file, '_'.join(a) + '2')
