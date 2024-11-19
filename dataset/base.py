import cv2
import numpy as np
from math import ceil
import glob
import os
import re

from torch.utils.data import Dataset
import logging
import pandas as pd

class BaseDataset(Dataset):
    def __init__(self, args, data_dirs=None, mode='train'):
        super().__init__()
        
        self.mode = mode
        self.args = args

        self.data_dirs = data_dirs
        self.processed_data_path = args.path['processed_data_path'] + '/' + self.args.dataset_name
        self.file_list_path = self.processed_data_path + '/' + args.path['file_list_name']
         
        # 如果mode是train，则processed_data_path中加入/train
        if mode == 'train':
            self.processed_data_path = self.processed_data_path + '/train'
            self.file_list_path = self.file_list_path.replace('file_list', 'file_list_train')
        else:
            self.processed_data_path = self.processed_data_path + '/test'
            self.file_list_path = self.file_list_path.replace('file_list', 'file_list_test')
                   
        self.dataset_name = args.dataset_name
        self.config_preprocess = args.config_preprocess
        
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

    def data_process(self):
        raise NotImplementedError("'data_process' Not Implemented")


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
    
    
    def preprocess(self,frames, bvp, config_preprocess):
        """Preprocesses the raw data. return frames_clips, bvps_clips"""

        frames = self.crop_face_resize(
            frames, 
            config_preprocess['Crop_Face']['Do_Crop_Face'], 
            config_preprocess['Crop_Face']['Backend'], 
            config_preprocess['Crop_Face']['Use_Large_Face_Box'], 
            config_preprocess['Crop_Face']['Large_Box_Coef'], 
            config_preprocess['Crop_Face']['Detection']['Do_Dynamic_Detection'], 
            config_preprocess['Crop_Face']['Detection']['Dynamic_Detection_Frequency'], 
            config_preprocess['Crop_Face']['Detection']['Use_Median_Face_Box'], 
            config_preprocess['Resize']['W'], 
            config_preprocess['Resize']['H'])
        data = list()
        for data_type in config_preprocess['Data_Type']:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(self.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(self.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)
        
        if config_preprocess['Label_Type'] == "Raw":
            pass
        elif config_preprocess['Label_Type'] == "DiffNormalized":
            bvp = self.diff_normalize_label(bvp)
        elif config_preprocess['Label_Type'] == "Standardized":
            bvp = self.standardized_label(bvp)
        else:
            raise ValueError("Unsupported label type!")
        
        if config_preprocess['Do_Chunk']:
            frames_clips, bvps_clips = self.chunk(data, bvp, config_preprocess['Chunk_Length'])

        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvp])
        
        return frames_clips, bvps_clips

    def face_detection(self, frame, backend, use_larger_box=False, larger_box_coef=1.0):
        """single frame face detection"""
        # use OpenCV's Haar cascade to detect face
        if backend == 'HC':
            cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
            detector = cv2.CascadeClassifier(cascade_path)
            # face_zone [x, y, w, h]
            face_zone = detector.detectMultiScale(frame)
            # if no face detected, return None
            if len(face_zone) == 0:
                self.logger.info('Error: No face detected!')
                detected_face_box = [0, 0, frame.shape[0], frame.shape[1]]
            elif len(face_zone) > 1:
                self.logger.info('More than one face detected! Only the largest face will be used.')
                # find the max width face
                max_width_index = np.argmax(face_zone[:, 2])
                detected_face_box = face_zone[max_width_index]
            else:
                detected_face_box = face_zone[0]
                
        # enlarge the box
        if use_larger_box:
            detected_face_box[0] = max(0, detected_face_box[0] - (larger_box_coef - 1.0) / 2 * detected_face_box[2])
            detected_face_box[1] = max(0, detected_face_box[1] - (larger_box_coef - 1.0) / 2 * detected_face_box[3])
            detected_face_box[2] = larger_box_coef * detected_face_box[2]
            detected_face_box[3] = larger_box_coef * detected_face_box[3]
        
        return detected_face_box

    # crop and resize face in frames
    def crop_face_resize(self, frames, use_face_detection, backend, use_larger_box, larger_box_coef, use_dynamic_detection,
                        detection_freq, use_median_box, width, height):
        # dynamic detection: 每隔 detection_freq 帧检测一次人脸
        if use_dynamic_detection:
            num_dynamic_det = ceil(frames.shape[0] / detection_freq) # 向上取整，为了覆盖所有的frame
        else:
            num_dynamic_det = 1
        face_region_all = []
        for idx in range(num_dynamic_det):
            if use_face_detection:
                face_region_all.append(self.face_detection(frames[idx * detection_freq], backend, use_larger_box, larger_box_coef))
            else:
                face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
        # convert to numpy array
        face_region_all = np.array(face_region_all, dtype='int')
        #! why median_box
        if use_median_box:
            face_region_median = np.median(face_region_all, axis=0).astype('int')
        
        # resize
        resized_frames = np.zeros((frames.shape[0], height, width, 3))
        for i in range(0,frames.shape[0]):
            frame = frames[i]
            if use_dynamic_detection:
                reference_index = i // detection_freq
            else:
                reference_index = 0
            
            if use_face_detection:
                if use_median_box:
                    face_region = face_region_median
                else:
                    face_region = face_region_all[reference_index]
                # resize frame
                frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                            max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
            resized_frames[i] = cv2.resize(frame, (width, height),interpolation=cv2.INTER_AREA)
        return resized_frames


    def chunk(self, data, label, chunk_length):
        """Chunk the data into clips."""

        clip_num = data.shape[0] // chunk_length
        frames_clips = [data[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        bvps_clips = [label[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)

    
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


    def diff_normalize_data(self,data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            # 计算 差分 
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        # 初始化一个全零数组，添加到最后一个位置
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        # 将所有的 nan 值替换为0
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data


    def diff_normalize_label(self,label):
        """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label


    def standardized_data(self,data):
        """Z-score standardization for video data."""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data


    def standardized_label(self,label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label
