import glob
import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.base import BaseDataset

class UbfcDataset(BaseDataset):
    def __init__(self, args, data_dirs=None, mode='train'):
        super().__init__(args, data_dirs, mode)
        '''
            -----------------
            RawData/
            |   |-- subject1/
            |       |-- vid.avi
            |       |-- ground_truth.txt
            |   |-- subject2/
            |       |-- vid.avi
            |       |-- ground_truth.txt
            |...
            |   |-- subjectn/
            |       |-- vid.avi
            |       |-- ground_truth.txt
            -----------------
        '''
    
    def data_process(self):
        
        data_dirs = self.data_dirs
        num_files = len(data_dirs)

        progress_bar = tqdm(list(range(num_files)))
    
        file_list = []
        for i in progress_bar:
            file_path = data_dirs[i]['path']
            progress_bar.set_description(f"Processing {file_path}")
            frames = self.read_video(file_path)
            self.logger.info('Processing file_path: ' + file_path)
            bvp = self.read_wave(file_path)
            self.logger.info('Read_video.shape: ' + str(frames.shape))
            self.logger.info('Read_wave.shape: ' + str(bvp.shape))
            # preprocess
            frames_clips, bvps_clips = self.preprocess(frames, bvp, self.config_preprocess)
            self.logger.info('Preprocessed_frames_clips.shape: ' + str(frames_clips.shape))
            self.logger.info('Preprocessed_bvps_clips.shape: ' + str(bvps_clips.shape))
            file_list += self.save(frames_clips, bvps_clips, data_dirs[i]['index'])
        file_list = pd.DataFrame(file_list, columns=['input_files'])
        
        if not os.path.exists(os.path.dirname(self.file_list_path)):
            os.makedirs(os.path.dirname(self.file_list_path))
        
        file_list.to_csv(self.file_list_path, index=False)
            
    @staticmethod
    def read_video(video_file):
            """读取视频文件，返回帧数据(T, H, W, 3)"""
            video_file = video_file + os.sep + "vid.avi"
            video_obj = cv2.VideoCapture(video_file)
            video_obj.set(cv2.CAP_PROP_POS_MSEC, 0)
            success, frame = video_obj.read()
            frames = list()
            while success:
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame)
                frames.append(frame[..., :3])
                success, frame = video_obj.read()
            frames = np.array(frames) 

            return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """读取 bvp 文件，返回 bvp 数据(T,)"""
        bvp_file = bvp_file + os.sep + "ground_truth.txt"
        with open(bvp_file, 'r') as f:
            str1 = f.read()
            str1 = str1.split('\n')
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)
    

class UbfcDataLoader():
    def __init__(self, preprocess_args, train_args):
        self.preprocess_args = preprocess_args
        self.train_args = train_args
    
    def get_raw_data(self,raw_data_path):
        data_dirs = glob.glob(raw_data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(raw_data_path + " data paths empty!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self,data_dirs, split_ratio, random_seed=42):
            if split_ratio == 1:
                return data_dirs, None
        
            if random_seed is not None:
                np.random.seed(random_seed)
            
            indices = np.random.permutation(len(data_dirs))
            split_point = int(split_ratio * len(data_dirs))
            
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]
            
            train_dirs = [data_dirs[i] for i in train_indices]
            test_dirs = [data_dirs[i] for i in test_indices]
            
            return train_dirs, test_dirs
    
         
    def get_dataloder(self):
    
        data_dirs = self.get_raw_data(self.preprocess_args.path['raw_dataset_path'])
        train_dirs, test_dirs = self.split_raw_data(data_dirs, self.preprocess_args.split_ratio)
        
        train_loader = DataLoader(
            UbfcDataset(self.preprocess_args, data_dirs=train_dirs, mode='train'),
            batch_size=self.train_args.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        test_loader = None
        if test_dirs is not None:
            test_loader = DataLoader(
                UbfcDataset(self.preprocess_args, data_dirs=test_dirs, mode='test'),
                batch_size=self.train_args.batch_size,
                shuffle=False,
                num_workers=4
            )
        
        return train_loader, test_loader
    
    