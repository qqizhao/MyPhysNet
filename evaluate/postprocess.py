import numpy as np
import scipy
import scipy.io
from scipy.signal import butter, periodogram, find_peaks
from scipy.sparse import spdiags

# need to be studied
def mag2db(mag):
    return 20. * np.log10(mag)


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def detrend(signal, lambda_value):
    """
    :param signal: T, or B x T
    :param lambda_value:
    :return:
    """
    T = signal.shape[-1]
    # observation matrix
    H = np.identity(T)  # T x T
    ones = np.ones(T)  # T,
    minus_twos = -2 * np.ones(T)  # T,
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (T - 2), T).toarray()
    designal = (H - np.linalg.inv(H + (lambda_value ** 2) * D.T.dot(D))).dot(signal.T).T
    return designal


def calculate_SNR(psd, freq, gtHR, target):
    """
    信噪比
    :param psd: predict PSD
    :param freq: predict frequency
    :param gtHR: ground truth
    :param target: signal type
    """
    gtHR = gtHR / 60
    gtmask1 = (freq >= gtHR - 0.1) & (freq <= gtHR + 0.1)
    gtmask2 = (freq >= gtHR * 2 - 0.1) & (freq <= gtHR * 2 + 0.1)
    sPower = psd[np.where(gtmask1 | gtmask2)].sum()
    if target == 'pulse':
        mask = (freq >= 0.75) & (freq <= 4)
    else:
        mask = (freq >= 0.08) & (freq <= 0.5)
    allPower = psd[np.where(mask)].sum()
    ret = mag2db(sPower / (allPower - sPower))
    return ret


# TODO: respiration 是否需要 cumsum; 短序列心率计算不准确
def fft_physiology(signal: np.ndarray, target="pulse", Fs=30, diff=True, detrend_flag=True):
    """
    利用 fft 计算 HR or FR
    get filter -> detrend -> get psd and freq -> get mask -> get HR
    
    Args:
        signal (np.ndarray): 输入信号,形状为 T 或 B x T
        target (str, optional): 目标信号类型,可选 'pulse' 或 'respiration'. Defaults to "pulse".
        Fs (int, optional): 采样频率. Defaults to 30.
        diff (bool, optional): 是否为差分信号. Defaults to True.
        detrend_flag (bool, optional): 是否需要去趋势. Defaults to True.

    Returns:
        np.ndarray: 计算得到的心率或呼吸率,单位为 bpm
    """
    # 如果是差分信号,需要先积分还原
    if diff:
        signal = signal.cumsum(axis=-1)
    # 去趋势
    if detrend_flag:
        signal = detrend(signal, 100)
    # 根据目标信号类型设计带通滤波器
    if target == "pulse":
        # 心率范围在 0.75Hz-2.5Hz (45-150 bpm)
        [b, a] = butter(1, [0.75 / Fs * 2, 2.5 / Fs * 2], btype='bandpass')
    else:
        # 呼吸率范围在 0.08Hz-0.5Hz (4.8-30 bpm)
        [b, a] = butter(1, [0.08 / Fs * 2, 0.5 / Fs * 2], btype='bandpass')
    # 带通滤波
    signal = scipy.signal.filtfilt(b, a, np.double(signal))
    # 计算功率谱密度
    N = next_power_of_2(signal.shape[-1])
    freq, psd = periodogram(signal, fs=Fs, nfft=N, detrend=False)
    # 根据目标信号类型设置频率掩码
    if target == "pulse":
        mask = np.argwhere((freq >= 0.75) & (freq <= 2.5))
    else:
        mask = np.argwhere((freq >= 0.08) & (freq <= 0.5))
    # 获取峰值对应的频率
    freq = freq[mask]
    if len(signal.shape) == 1:
        # 单个信号
        idx = psd[mask.reshape(-1)].argmax(-1)
    else:
        # 批量信号
        idx = psd[:, mask.reshape(-1)].argmax(-1)
    # 将频率转换为 bpm
    phys = freq[idx] * 60
    return phys.reshape(-1)


def peak_physiology(signal: np.ndarray, target="pulse", Fs=30, diff=True, detrend_flag=True):
    """
    利用 ibi 计算 HR or FR
    get filter -> detrend -> get psd and freq -> get mask -> get HR
    :param signal: T, or B x T
    :param target: pulse or respiration
    :param Fs:
    :param diff: 是否为差分信号
    :param detrend_flag: 是否需要 detrend
    :return:
    """
    if diff:
        signal = signal.cumsum(axis=-1)
    if detrend_flag:
        signal = detrend(signal, 100)
    if target == 'pulse':
        [b, a] = butter(1, [0.75 / Fs * 2, 2.5 / Fs * 2],
                        btype='bandpass')  # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / Fs * 2, 0.5 / Fs * 2], btype='bandpass')
    # bandpass
    signal = scipy.signal.filtfilt(b, a, np.double(signal))
    T = signal.shape[-1]
    signal = signal.reshape(-1, T)
    phys = []
    for s in signal:
        peaks, _ = find_peaks(s)
        phys.append(60 * Fs / np.diff(peaks).mean(axis=-1))

    return np.asarray(phys)
