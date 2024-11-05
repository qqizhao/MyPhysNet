import glob
import os
import re
import cv2
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import pandas as pd
import logging
from tqdm.auto import tqdm

from torch.utils.data import Dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import PreprocessArgs
from data_process.utils import *


class MyDataset(Dataset):
    def __init__(self, args):
        super(MyDataset, self).__init__()
        
        self.args = args
        self.raw_dataset_path = args.path['raw_dataset_path']
        self.processed_data_path = args.path['processed_data_path']
        self.file_list_path = args.path['file_list_path']
        self.dataset_name = args.dataset_name
        
        self.config_preprocess = args.config_preprocess
        
        # 通过名称得到日志logger
        self.logger = logging.getLogger('MyLogger')
        
        # npy 文件路径
        self.inputs = list()
        self.labels = list()
        
        if args.do_preprocess:
            self.data_process()
            
        self.load_preprocessed_data()
        self.logger.info('Processed Data Path: ' + self.processed_data_path)
        self.logger.info('File List Path: ' + self.file_list_path)
        self.logger.info(f"{self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}")
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        data = np.load(self.inputs[idx])
        label = np.load(self.labels[idx])
        
        # turn data(depth, height, width, channels) to (depth, channels, height, width)
        data = np.transpose(data, (0, 3, 1, 2))
        data = np.float32(data)
        label = np.float32(label)
        
        item_path = self.inputs[idx]
        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.index('_')
        file_name = item_path_filename[:split_idx]
        chunk_idx = item_path_filename[split_idx + 6:].split('.')[0]
        
        return data, label, file_name, chunk_idx


    def load_preprocessed_data(self):
        """读取预处理后的数据"""        

        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in inputs]

        self.inputs = inputs
        self.labels = labels
    
        self.preprocessed_data_len = len(inputs)
    
    def data_process(self):
        
        data_dirs = get_raw_data(self.raw_dataset_path)
        data_dirs = split_raw_data(data_dirs, self.args.begin, self.args.end)
        num_files = len(data_dirs)
        progress_bar = tqdm(list(range(num_files)))
    
        file_list = []
        for i in progress_bar:
            file_path = data_dirs[i]['path']
            progress_bar.set_description(f"Processing {file_path}")
            frames = read_video(file_path)
            self.logger.info('process file_path: ' + file_path)
            bvp = read_wave(file_path)
            self.logger.info('read_video.shape: ' + str(frames.shape))
            self.logger.info('read_wave.shape: ' + str(bvp.shape))
            # preprocess
            frames_clips, bvps_clips = preprocess(frames, bvp, self.config_preprocess)
            self.logger.info('preprocessed_frames_clips.shape: ' + str(frames_clips.shape))
            self.logger.info('preprocessed_bvps_clips.shape: ' + str(bvps_clips.shape))
            file_list += self.save(frames_clips, bvps_clips, data_dirs[i]['index'])
        file_list = pd.DataFrame(file_list, columns=['input_files'])
        
        if not os.path.exists(os.path.dirname(self.file_list_path)):
            os.makedirs(os.path.dirname(self.file_list_path))
        
        file_list.to_csv(self.file_list_path, index=False)
            
    
    def save(self,frames_clips, bvps_clips, filename):
        """生成file_list"""
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        count = 0
        file_list = []
        for i in range(len(frames_clips)):
            input_path_name = self.processed_data_path + os.sep + f"{filename}_input{count}.npy"
            label_path_name = self.processed_data_path + os.sep + f"{filename}_label{count}.npy"
            file_list.append(input_path_name)
            np.save(input_path_name, frames_clips[i].astype(np.float32))
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return file_list

