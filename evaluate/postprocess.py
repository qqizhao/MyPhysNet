import numpy as np
import scipy
import scipy.io
from scipy.signal import butter, periodogram, find_peaks
from scipy.sparse import spdiags


def mag2db(mag):
    return 20. * np.log10(mag)


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def detrend(input_signal, lambda_value):
    """去除信号中的长期趋势
    Args:
        input_signal: 输入信号
        lambda_value: 平滑参数,控制去趋势的强度
    Returns:
        detrended_signal: 去除趋势后的信号
    """
    T = input_signal.shape[0]
    
    H = np.identity(T)   # 单位矩阵
    ones = np.ones(T)
    minus_twos = -2 * np.ones(T)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, T - 2, T).toarray()  # 差分矩阵

    # Apply smoothing
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal
    )
    return detrended_signal


def calculate_SNR(psd, ppg_freq, heart_rate, target):
    """ 计算信噪比
    Args:
        psd (np.ndarray): 预测功率谱密度
        freq (np.ndarray): 预测频率
        gtHR (np.ndarray): 真实心率
        target (str): 信号类型
    Returns:
        SNR (float): 信噪比
    """
    heart_rate_hz = heart_rate / 60
    gtmask1 = (ppg_freq >= heart_rate_hz - 0.1) & (ppg_freq <= heart_rate_hz + 0.1)
    gtmask2 = (ppg_freq >= heart_rate_hz * 2 - 0.1) & (ppg_freq <= heart_rate_hz * 2 + 0.1)
    sPower = psd[np.where(gtmask1 | gtmask2)].sum()
    if target == 'pulse':
        mask = (ppg_freq >= 0.75) & (ppg_freq <= 4)
    else:
        mask = (ppg_freq >= 0.08) & (ppg_freq <= 0.5)
    allPower = psd[np.where(mask)].sum()
    SNR = mag2db(sPower / (allPower - sPower))
    return SNR


def calculate_HR(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = detrend(np.cumsum(predictions), 100)
        labels = detrend(np.cumsum(labels), 100)
    else:
        predictions = detrend(predictions, 100)
        labels = detrend(labels, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz  -> [45, 150]
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    
    # macc = _compute_macc(predictions, labels)

    if hr_method == 'FFT':
        hr_pred = fft_hr(predictions, fs=fs)
        hr_label = fft_hr(labels, fs=fs)
    elif hr_method == 'Peak':
        hr_pred = peak_hr(predictions, fs=fs)
        hr_label = peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    # SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    return hr_pred, hr_label


def fft_hr(signal, fs=30):
    """ 利用 fft 计算 HR or FR """

    N = next_power_of_2(signal.shape[-1])
    ppg_freq, ppg_pow = periodogram(signal, fs=fs, nfft=N, detrend=False)
    
    mask = np.argwhere((ppg_freq >= 0.75) & (ppg_freq <= 2.5))
    ppg_freq = ppg_freq[mask]
    if len(signal.shape) == 1:
        idx = ppg_pow[mask.reshape(-1)].argmax(-1)
    else:
        idx = ppg_pow[:, mask.reshape(-1)].argmax(-1)
    dominant_frequency = ppg_freq[idx] * 60
    return dominant_frequency.reshape(-1)


def peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

    # T = signal.shape[-1]
    # signal = signal.reshape(-1, T)
    # phys = []
    # for s in signal:
    #     peaks, _ = find_peaks(s)
    #     phys.append(60 * Fs / np.diff(peaks).mean(axis=-1))

    # return np.asarray(phys)