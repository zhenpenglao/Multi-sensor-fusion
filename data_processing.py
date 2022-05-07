# Data science libraries
import copy

import numpy as np
import random
from pathlib import Path
from data_loader import get_df_all

working_dir = Path('.')
DATA_PATH = Path("./data/DDS/bearingset/G1")
save_model_path = working_dir / 'Model'
data_cat = ['motor_vibration', 'planetary_vibration_x', 'planetary_vibration_y', 'planetary_vibration_z',
           'motor_torque', 'parallel_vibration_x', 'parallel_vibration_y', 'parallel_vibration_z']
Segment_length = 4096
num_channel = 3


def sequence_to_images(data):
    group_imgs, label = [], []
    for num in range(int(data.shape[0]/3)):
        X_raw = data.iloc[num_channel * num: num_channel * (num + 1), 2:].T
        X_label = data.iloc[num_channel * num, 0].T
        X_chl = np.array(X_raw)
        part_imgs = []
        for channel in range(3):
            # 将4096个数据点转化成64×64的二维图
            temp = X_chl[0:4096, channel]
            temp = temp.reshape(64, 64)
            max = np.max(temp)
            min = np.min(temp)
            for i in range(64):
                for j in range(64):
                    temp[i][j] = 255 * (temp[i][j] - min) / (max - min)
            part_imgs.append(copy.deepcopy(temp))
        imgs = np.dstack((part_imgs[0], part_imgs[1], part_imgs[2]))
        group_imgs.append(imgs)
        label.append(X_label)
    return group_imgs, label


dataset, label = [], []
df_all = get_df_all(DATA_PATH, data_cat=data_cat, segment_length=Segment_length, normalize=True)
group_imgs, group_label = sequence_to_images(df_all)
dataset.extend(group_imgs)
label.extend(group_label)
np.savez("data/dataset", *dataset)
np.savez("data/label", *label)

print("data processing done!")