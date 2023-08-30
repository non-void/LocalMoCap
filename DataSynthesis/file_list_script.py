import os
import numpy as np

dataset_name = 'CMU'
dataset_path = os.path.join('/media/zheng/0258541058540537/Dataset', dataset_name)
file_folder = []
file_name_list = []

for folder_name in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, file)):
                file_folder += [folder_name]
                file_name_list += [file]

np.savez(os.path.join('data', f'{dataset_name}_file_list.npz'), file_folder=file_folder, file_name_list=file_name_list)
