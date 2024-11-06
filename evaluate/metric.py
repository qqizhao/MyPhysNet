import torch
import numpy as np


def cal_metric(pred_phys: np.ndarray, label_phys: np.ndarray,
               methods=None) -> list:
    """计算预测值和真实值之间的评价指标

    Args:
        pred_phys (np.ndarray): 预测值
        label_phys (np.ndarray): 真实值 
        methods (list, optional): 需要计算的指标列表. Defaults to None.

    Returns:
        list: 返回计算得到的各项指标值列表
    """
    pred_phys = np.asarray(pred_phys)
    label_phys = np.asarray(label_phys)
    
    if methods is None:
        methods = ["MAE", "RMSE", "MAPE", "R"]
    pred_phys = pred_phys.reshape(-1)  # 展平为一维数组
    print('pred_phys: ', pred_phys)
    label_phys = label_phys.reshape(-1)
    print('label_phys: ', label_phys)
    ret = [] * len(methods)
    for m in methods:
        if m == "MAE":  
            ret.append(np.abs(pred_phys - label_phys).mean())
        elif m == "RMSE":  
            ret.append(np.sqrt((np.square(pred_phys - label_phys)).mean()))
        elif m == "MAPE":  # 平均绝对百分比误差
            ret.append((np.abs((pred_phys - label_phys) / label_phys)).mean() * 100)
        elif m == "R":  
            temp = np.corrcoef(pred_phys, label_phys)
            if np.isnan(temp).any() or np.isinf(temp).any():
                ret.append(-1)
            else:
                ret.append(float(temp[0, 1]))

    return ret

