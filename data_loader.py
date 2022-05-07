# Helper functions to read and preprocess data files from Matlab format
# Data science libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Others
from pathlib import Path
from tqdm.auto import tqdm
import requests
from PCA import pca

be_path = 'data/DDS/bearingset'
ge_path = 'data/DDS/gearset'

# csv_data = pd.read_csv(be_path+'ball_20_0.csv')  # 读取训练数据
# print(csv_data.shape)


def csvfile_to_read(folder_path):
    '''
    Read all the csv files of the DDS Gearbox Dataset and return a
    dictionary.

    Parameter:
        folder_path:
            Path (Path object) of the folder which contains the csv files.
    Return:
        output_dic:
            Dictionary which contains data of all files in the folder_path.
    '''
    output_dic = {}
    files = Path(folder_path).glob('*.csv')
    for filepath in files:
        # strip the folder path and get the filename only.
        key_name = str(filepath).split('\\')[-1]
        output_dic[key_name] = pd.read_csv(filepath)

    data_dic_extract, idx = {}, 0
    for k, v in output_dic.items():
        v.drop(range(15), inplace=True)
        v = v.reset_index(drop=True)
        data_dic_extract[idx] = {
            # df.iloc[], 0取第一列，-1取最后一列，：取所有列
            'filename': k,
            'motor_vibration': np.array(v.iloc[:, 0]).astype('float'),
            'planetary_vibration_x': np.array(v.iloc[:, 1]).astype('float'),
            'planetary_vibration_y': np.array(v.iloc[:, 2]).astype('float'),
            'planetary_vibration_z': np.array(v.iloc[:, 3]).astype('float'),
            'motor_torque': np.array(v.iloc[:, 4]).astype('float'),
            'parallel_vibration_x': np.array(v.iloc[:, 5]).astype('float'),
            'parallel_vibration_y': np.array(v.iloc[:, 6]).astype('float'),
            'parallel_vibration_z': np.array(v.iloc[:, 7]).astype('float'),
        }
        idx += 1
    return data_dic_extract


def label(filename):
    '''
    Function to create label for each signal based on the filename. Apply this
    to the "filename" column of the DataFrame.
    Usage:
        df['label'] = df['filename'].apply(label)
    '''
    if 'health' in filename:
        return 'NO'
    elif 'ball' in filename:
        return 'BF'
    elif 'inner' in filename:
        return 'IRF'
    elif 'outer' in filename:
        return 'ORF'
    elif 'comb' in filename:
        return 'IOF'


def csv_to_df(folder_path, data_cat):
    '''
    Read all the csv files in the folder, preprocess, and return a DataFrame
    with data specified by data_cat

    Parameter:
        folder_path:
            Path (Path object) of the folder which contains the csv files.
        data_cat:
            Data category of interest, i.e. 'motor_vibration', 'planetary_vibration_x', 'planetary_vibration_y',
                                            'planetary_vibration_z', 'motor_torque', 'parallel_vibration_x',
                                            'parallel_vibration_y', 'parallel_vibration_z'
    Return:
        DataFrame with preprocessed data
    '''
    dic = csvfile_to_read(folder_path)
    # 将dict转换为DataFrame对象
    df = pd.DataFrame.from_dict(dic).T

    df['label'] = df['filename'].apply(label)
    if type(data_cat) is not list:
        return df[['filename', data_cat, 'label']]
    elif type(data_cat) is list:
        return df[['filename', *data_cat, 'label']]


def split_signal(df_split, data_cat):
    '''
    This function divide the signal into segments, each with a specific number
    of points as defined by segment_length. Each segment will be added as an
    example (a row) in the returned DataFrame. Thus it increases the number of
    training examples. The remaining points which are less than segment_length
    are discarded.

    Parameter:
        df:
            DataFrame returned by csv_to_df()
        segment_length:
            Number of points per segment.
    Return:
        DataFrame with segmented signals and their corresponding filename and label
    '''
    if type(data_cat) is not list:
        dic = {}
        idx = 0
        for i in range(df_split.shape[0]):
            n_sample_points = len(df_split.iloc[i, 1])
            dic[idx] = {
                'signal': df_split.iloc[i, 1],
                'label': df_split.iloc[i, 2],
                'filename': df_split.iloc[i, 0]
            }
            idx += 1
        df_tmp = pd.DataFrame.from_dict(dic, orient='index')
        df_output = pd.concat(
            [df_tmp[['label', 'filename']],
             pd.DataFrame(np.vstack(df_tmp["signal"]))
             ],
            axis=1)
        return df_output

    elif type(data_cat) is list:
        dic, idx = {}, 0
        for i in range(df_split.shape[0]):
            for category in data_cat:
                dic[idx] = {
                    # df.iloc[], 0取第一列，-1取最后一列，：取所有列
                    'signal': df_split.iloc[i, :][category],
                    'label': df_split.iloc[i, -1],
                    'filename': df_split.iloc[i, 0]
                }
                idx += 1
        df_tmp = pd.DataFrame.from_dict(dic, orient='index')
        # 根据索引将数据进行拼接
        df_split = pd.concat(
            [df_tmp[['label', 'filename']],
             # np.vstack() 垂直(行)按顺序堆叠数组
             pd.DataFrame(np.vstack(df_tmp["signal"]))
             ], axis=1)
        return df_split


def divide_signal(df_split, segment_length, data_cat):
    '''
    This function divide the signal into segments, each with a specific number
    of points as defined by segment_length. Each segment will be added as an
    example (a row) in the returned DataFrame. Thus it increases the number of
    training examples. The remaining points which are less than segment_length
    are discarded.

    Parameter:
        df:
            DataFrame returned by matfile_to_df()
        segment_length:
            Number of points per segment.
    Return:
        DataFrame with segmented signals and their corresponding filename and label
    '''
    if type(data_cat) is not list:
        dic, idx = {}, 0
        for i in range(df_split.shape[0]):
            n_sample_points = len(df_split.iloc[i, 1])
            # n_segments = n_sample_points / n_sample_points
            n_segments = n_sample_points // segment_length
            for segment in range(n_segments):
                dic[idx] = {
                    'signal': df_split.iloc[i, 1][segment_length * segment:segment_length * (segment + 1)],
                    'label': df_split.iloc[i, 2],
                    'filename': df_split.iloc[i, 0]
                }
                idx += 1
        df_tmp = pd.DataFrame.from_dict(dic, orient='index')
        df_output = pd.concat(
            [df_tmp[['label', 'filename']],
             pd.DataFrame(np.vstack(df_tmp["signal"]))
             ],
            axis=1)
        return df_output

    elif type(data_cat) is list:
        dic, idx = {}, 0
        num_channel = 3
        n_sample_points = df_split.shape[1] - 2
        n_segments = n_sample_points // segment_length
        for i in range(int(df_split.shape[0] / num_channel)):
            for segment in range(1000):
                dic[idx] = {
                    'label': df_split.iloc[num_channel*i:num_channel*(i+1), 0],
                    'filename': df_split.iloc[num_channel*i:num_channel*(i+1), 1],
                    'signal': df_split.iloc[num_channel*i:num_channel*(i+1), 1045*segment+2:1045*segment+segment_length+2]
                }
                idx += 1
        df_tmp = pd.DataFrame.from_dict(dic, orient='index')
        df_output = pd.concat(
            [pd.DataFrame(np.hstack(df_tmp["label"]), columns=["label"]),
             pd.DataFrame(np.hstack(df_tmp["filename"]), columns=["filename"]),
             pd.DataFrame(np.vstack(df_tmp["signal"]))
             ], axis=1)
        return df_output


def reduce_signal(df_split, data_cat):
    dic, idx = {}, 0
    num_sensor = len(data_cat)
    for i in range(int(df_split.shape[0] / num_sensor)):
        X_rce = pca(df_split.iloc[num_sensor * i: num_sensor * (i + 1), 2:].T, n_components=3)
        dic[idx] = {
            'label': df_split.iloc[num_sensor * i:num_sensor * (i + 1):3, 0],
            'filename': df_split.iloc[num_sensor * i:num_sensor * (i + 1):3, 1],
            'signal': X_rce.T
        }
        idx += 1
    df_tmp = pd.DataFrame.from_dict(dic, orient='index')
    df_reduce = pd.concat(
        [pd.DataFrame(np.hstack(df_tmp["label"]), columns=["label"]),
         pd.DataFrame(np.hstack(df_tmp["filename"]), columns=["filename"]),
         pd.DataFrame(np.vstack(df_tmp["signal"]))
         ], axis=1)
    return df_reduce


def normalize_signal(df, data_cat):
    '''
    Normalize the signals in the DataFrame returned by csv_to_df() by subtracting
    the mean and dividing by the standard deviation.
    '''
    if type(data_cat) is not list:
        mean = df[data_cat].apply(np.mean)
        std = df[data_cat].apply(np.std)
        df[data_cat] = (df[data_cat] - mean) / std
    elif type(data_cat) is list:
        for category in data_cat:
            mean = df[category].apply(np.mean)
            std = df[category].apply(np.std)
            df[category] = (df[category] - mean) / std
    return df


def get_df_all(data_path, data_cat, segment_length=512, normalize=False):
    '''
    Load, preprocess and return a DataFrame which contains all signals data and
    labels and is ready to be used for model training.

    Parameter:
        normal_path:
            Path of the folder which contains csv files of normal bearings
        DE_path:
            Path of the folder which contains csv files of DE faulty bearings
        segment_length:
            Number of points per segment. See divide_signal() function
        normalize:
            Boolean to perform normalization to the signal data
        data_cat:
            Data category of interest, i.e. 'motor_vibration', 'planetary_vibration_x', 'planetary_vibration_y',
                                            'planetary_vibration_z', 'motor_torque', 'parallel_vibration_x',
                                            'parallel_vibration_y', 'parallel_vibration_z'
    Return:
        df_all:
            DataFrame which is ready to be used for model training.
    '''

    df = csv_to_df(data_path, data_cat)

    # if normalize:
    #     normalize_signal(df, data_cat)
    df_split = split_signal(df, data_cat)
    df_reduce = reduce_signal(df_split, data_cat)
    df_processed = divide_signal(df_reduce, segment_length, data_cat)

    map_label = {'NO': 0, 'BF': 1, 'IRF': 2, 'ORF': 3, 'IOF': 4}
    # 将对应的标签映射为数字
    df_processed['label'] = df_processed['label'].map(map_label)
    return df_processed
