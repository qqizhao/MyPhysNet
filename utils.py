import logging
import random
import os
import numpy as np
import torch


def set_random_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu
    # torch.backends.cudnn.deterministic = True  # 会大大降低速度
    torch.backends.cudnn.benchmark = True  # False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True  # 增加运行效率，默认就是True
    torch.manual_seed(seed)
    
    
def create_logger(exp_dir):
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
        
    logging.basicConfig(

        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
        filename=f'{exp_dir}/logger.log',  # 设置日志文件名
        filemode='a'  # 追加模式
    )
    logger = logging.getLogger('MyLogger')
    
    # Create a StreamHandler to output logs to the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(stream_handler)
    
    return logger

def create_exp_dir(logs_dir='./logs'):
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    existing_dirs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
    exp_numbers = [int(d[3:]) for d in existing_dirs if d.startswith('exp') and d[3:].isdigit()]
    next_exp_number = max(exp_numbers, default=0) + 1

    new_exp_dir = os.path.join(logs_dir, f'exp{next_exp_number}')
    os.makedirs(new_exp_dir)
    
    return new_exp_dir

def get_max_exp(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    existing_dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    
    exp_numbers = [int(d[3:]) for d in existing_dirs if d.startswith('exp') and d[3:].isdigit()]
    max_exp_number = max(exp_numbers, default=0) 
    max_exp_dir = os.path.join(dir, f'exp{max_exp_number}')
    
    return max_exp_dir

def get_latest_checkpoint(exp_dir):
    
    # 获取所有.pth文件
    checkpoints = [f for f in os.listdir(exp_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None
        
    # 按文件名排序并返回最后一个
    latest_checkpoint = sorted(checkpoints)[-1]
    return os.path.join(exp_dir, latest_checkpoint)


def merge_clips(x):
    """将属于同一个subject的clips合并"""

    # 按chunk索引排序
    sort_x = sorted(x.items(), key=lambda x: x[0])
    sort_x = [i[1] for i in sort_x]
    sort_x = np.concatenate(sort_x, axis=0)
    # 展平为一维数组并返回
    return sort_x.reshape(-1)