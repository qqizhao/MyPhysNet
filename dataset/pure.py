import glob
import os
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.base import BaseDataset


class PureDataset(BaseDataset):
    def __init__(self, args, data_dirs=None, mode='train'):
        super().__init__(args, data_dirs, mode)
        '''
            -----------------
            RawData/
            |   |-- 01-01/
            |      |-- 01-01/
            |      |-- 01-01.json
            |   |-- 01-02/
            |      |-- 01-02/
            |      |-- 01-02.json
            |...
            |   |-- ii-jj/
            |      |-- ii-jj/
            |      |-- ii-jj.json
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
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        for dir in os.listdir(video_file):
            if os.path.isdir(os.path.join(video_file, dir)):
                video_file = os.path.join(video_file, dir)
        all_png = sorted(glob.glob(os.path.join(video_file, '*.png')))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        # 找到 bvp file 下的 json文件
        for file in os.listdir(bvp_file):
            if file.endswith('.json'):
                bvp_file = os.path.join(bvp_file, file)
                break
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["waveform"]
                     for label in labels["/FullPackage"]]
        return np.asarray(waves)
    

class PureDataLoader():
    def __init__(self, preprocess_args, train_args):
        self.preprocess_args = preprocess_args
        self.train_args = train_args
    
    def get_raw_data(self,raw_data_path):
        data_dirs = glob.glob(raw_data_path + os.sep + "*-*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('-', '')
            index = int(subject_trail_val)
            subject = int(subject_trail_val[0:2])
            dirs.append({"index": index, "path": data_dir, "subject": subject})
            
        return dirs


    def split_raw_data(self,data_dirs, split_ratio):
        
        # return the full directory
        if split_ratio == 1:
            return data_dirs, None

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})
        
        subj_list = list(data_info.keys())  # all subjects by number ID (1-27)
        num_subjs = len(subj_list)  # number of unique subjects
        subj_train = list(range(0,int(split_ratio * num_subjs)))
        subj_test = list(range(int(split_ratio * num_subjs), num_subjs))

        train_dirs = []
        test_dirs = []
        
        for i in subj_train:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            train_dirs += subj_files  # add file information to file_list (tuple of fname, subj ID, trial num,
        
        for i in subj_test:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            test_dirs += subj_files  # add file information to file_list (tuple of fname, subj ID, trial num,

        return train_dirs, test_dirs
    
         
    def get_dataloder(self):
    
        data_dirs = self.get_raw_data(self.preprocess_args.path['raw_dataset_path'])
        train_dirs, test_dirs = self.split_raw_data(data_dirs, self.preprocess_args.split_ratio)
        
        train_loader = DataLoader(
            PureDataset(self.preprocess_args, data_dirs=train_dirs, mode='train'),
            batch_size=self.train_args.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        test_loader = None
        if test_dirs is not None:
            test_loader = DataLoader(
                PureDataset(self.preprocess_args, data_dirs=test_dirs, mode='test'),
                batch_size=self.train_args.batch_size,
                shuffle=False,
                num_workers=4
            )
        
        return train_loader, test_loader
    
    