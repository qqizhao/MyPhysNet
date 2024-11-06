import cv2
import numpy as np
from math import ceil
import glob
import os
import re


def preprocess(frames, bvp, config_preprocess):
    """Preprocesses the raw data. return frames_clips, bvps_clips"""

    frames = crop_face_resize(
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
            data.append(diff_normalize_data(f_c))
        elif data_type == "Standardized":
            data.append(standardized_data(f_c))
        else:
            raise ValueError("Unsupported data type!")
    data = np.concatenate(data, axis=-1)
    
    if config_preprocess['Label_Type'] == "Raw":
        pass
    elif config_preprocess['Label_Type'] == "DiffNormalized":
        bvp = diff_normalize_label(bvp)
    elif config_preprocess['Label_Type'] == "Standardized":
        bvp = standardized_label(bvp)
    else:
        raise ValueError("Unsupported label type!")
    
    if config_preprocess['Do_Chunk']:
        frames_clips, bvps_clips = chunk(data, bvp, config_preprocess['Chunk_Length'])

    else:
        frames_clips = np.array([data])
        bvps_clips = np.array([bvp])
    
    return frames_clips, bvps_clips


def face_detection(frame, backend, use_larger_box=False, larger_box_coef=1.0):
    """single frame face detection"""
    # use OpenCV's Haar cascade to detect face
    if backend == 'HC':
        cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        detector = cv2.CascadeClassifier(cascade_path)
        # face_zone [x, y, w, h]
        face_zone = detector.detectMultiScale(frame)
        # if no face detected, return None
        if len(face_zone) == 0:
            print('Error: No face detected!')
            detected_face_box = [0, 0, frame.shape[0], frame.shape[1]]
        elif len(face_zone) > 1:
            print('More than one face detected! Only the largest face will be used.')
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
def crop_face_resize(frames, use_face_detection, backend, use_larger_box, larger_box_coef, use_dynamic_detection,
                    detection_freq, use_median_box, width, height):
    # dynamic detection: 每隔 detection_freq 帧检测一次人脸
    if use_dynamic_detection:
        num_dynamic_det = ceil(frames.shape[0] / detection_freq) # 向上取整，为了覆盖所有的frame
    else:
        num_dynamic_det = 1
    face_region_all = []
    for idx in range(num_dynamic_det):
        if use_face_detection:
            face_region_all.append(face_detection(frames[idx * detection_freq], backend, use_larger_box, larger_box_coef))
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


def chunk(data, label, chunk_length):
    """Chunk the data into clips."""

    clip_num = data.shape[0] // chunk_length
    frames_clips = [data[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
    bvps_clips = [label[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
    return np.array(frames_clips), np.array(bvps_clips)


def get_raw_data(raw_data_path):
        """获取原始数据"""
        data_dirs = glob.glob(raw_data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(raw_data_path + " data paths empty!")
        dirs = [{"index": re.search(
            'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

def split_raw_data(data_dirs, split_ratio, random_seed=42):
        """随机分割数据"""
        if split_ratio == 1:
            return data_dirs, None
    
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 随机打乱索引
        indices = np.random.permutation(len(data_dirs))
        split_point = int(split_ratio * len(data_dirs))
        
        # 使用随机索引分割数据
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        
        train_dirs = [data_dirs[i] for i in train_indices]
        test_dirs = [data_dirs[i] for i in test_indices]
        
        return train_dirs, test_dirs


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

def read_wave(bvp_file):
    """读取 bvp 文件，返回 bvp 数据(T,)"""
    bvp_file = bvp_file + os.sep + "ground_truth.txt"
    with open(bvp_file, 'r') as f:
        str1 = f.read()
        str1 = str1.split('\n')
        bvp = [float(x) for x in str1[0].split()]
    return np.asarray(bvp)
    

def diff_normalize_data(data):
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


def diff_normalize_label(label):
    """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
    diff_label = np.diff(label, axis=0)
    diffnormalized_label = diff_label / np.std(diff_label)
    diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
    diffnormalized_label[np.isnan(diffnormalized_label)] = 0
    return diffnormalized_label


def standardized_data(data):
    """Z-score standardization for video data."""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data


def standardized_label(label):
    """Z-score standardization for label signal."""
    label = label - np.mean(label)
    label = label / np.std(label)
    label[np.isnan(label)] = 0
    return label
