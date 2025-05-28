import os
import cv2
import numpy as np

import torch
import librosa
import soundfile
import resampy
from PIL import Image

def _read_img_lmdb(env, key, size):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img

def read_img(env, path, istrain = False, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
#         print(path)
        #img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        

        img = cv2.imread(path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not istrain:
            img = img.resize((256, 256))
            #img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        img = np.array(img)
    else:
        img = _read_img_lmdb(env, path, size)
#     print(img.shape)
#     if img is None:
#         print(path)
#     print(img.shape)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def stft_remove(data, n_fft = 522, hop_length = 250):
    window = torch.hann_window(n_fft).cuda()
    tmp = torch.stft(data, n_fft, hop_length, window = window, return_complex=True)
    amplitude = torch.abs(tmp)
    phase = torch.angle(tmp)

    return amplitude, phase

def istft_remove(amplitude, phase, n_fft = 522, hop_length = 250):
    window = torch.hann_window(n_fft).cuda()
    # amplitude와 phase를 사용하여 복소수 STFT 복원
    tmp = amplitude * torch.exp(1j * phase)
    # ISTFT를 사용하여 원래 신호로 복원

    return torch.istft(tmp, n_fft, hop_length=hop_length, window=window, return_complex=False)

def stft(data, n_fft = 510, hop_length = 250):
    window = torch.hann_window(n_fft).cuda()
    tmp = torch.stft(data, n_fft=n_fft, hop_length=hop_length, window = window, return_complex=True)
    tmp = torch.view_as_real(tmp)
    return tmp

def istft(data, n_fft = 510, hop_length = 250):
    window = torch.hann_window(n_fft).cuda()
    imsi = torch.istft(torch.view_as_complex(data), n_fft=n_fft, hop_length = hop_length, window = window, return_complex=False)
    
    return imsi

def quantize_feature_map(feature_map):
    """
    입력된 피처 맵 텐서를 8비트(qint8)로 양자화합니다.

    Args:
        feature_map (torch.Tensor): 양자화할 피처 맵 텐서 (float32 타입)

    Returns:
        torch.Tensor: 양자화된 피처 맵 텐서 (qint8 타입)
    """
    # 스케일과 제로 포인트 계산 (스칼라 값으로 변환)
    scale = feature_map.abs().max().item() / 127
    zero_point = 0  # 대칭 양자화이므로 0으로 설정

    # 텐서 양자화
    q_feature_map = torch.quantize_per_tensor(feature_map, scale, zero_point, dtype=torch.qint8)

    return q_feature_map