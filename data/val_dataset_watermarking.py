'''
Vimeo90K dataset
support reading images from lmdb, image folder and memcached
'''
import logging
import os
import cv2
import face_recognition
import os.path as osp
import librosa
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import data.util as util
from moviepy.editor import VideoFileClip

logger = logging.getLogger('base')

class TestDataset(data.Dataset):
    '''
    Reading the training Vimeo90K dataset
    key example: 00001_0001 (_1, ..., _7)
    GT (Ground-Truth): 4th frame;
    LQ (Low-Quality): support reading N LQ frames, N = 1, 3, 5, 7 centered with 4th frame
    '''

    def __init__(self, opt):
        super(TestDataset, self).__init__()
        self.opt = opt
        self.txt_path = self.opt['txt_path']
        self.list_video = []
        self.fps = self.opt['fps']
        self.audio_sample_rate = self.opt['audio_sample_rate']
        self.audio_sample = self.audio_sample_rate // self.fps

        with open(self.txt_path, 'r') as f:
            self.dir_list = [a.strip() for a in f.readlines()]

    def __getitem__(self, index):
        GT_size = self.opt['GT_size']
        path = self.dir_list[index]
        
        video_path = [p for p in sorted(os.listdir(path))]
        audio_path = os.path.join(path.replace('frames', 'audio_only'), 'audio.wav')

        frames = []
        for p in video_path:   
            path2 = os.path.join(path, p) 
            frames.append(util.read_img(None, path2))

        frames_np = np.array(frames)
        n_frames = frames_np.shape[0]

        audio, sr = librosa.load(audio_path, sr=None)
        
        
        if len(audio.shape) == 2:
            audio = audio[:, 0]

        # Resample to 4000 Hz
        if sr != self.audio_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_sample_rate)

        audio_sample_per_frame = self.audio_sample_rate // self.fps
        audio_chunks = [
            audio[(i)*audio_sample_per_frame:(i+1) * audio_sample_per_frame] 
            for i in range(n_frames)
        ]

        # 마지막 청크가 충분한 길이가 아니라면 패딩
        if len(audio_chunks[-1]) < audio_sample_per_frame:
            pad_length = audio_sample_per_frame - len(audio_chunks[-1])
            audio_chunks[-1] = np.pad(audio_chunks[-1], (0, pad_length), mode='constant')

        audio = np.array(audio_chunks, dtype=np.float32) 

        # 마지막 청크가 충분한 길이가 아니라면 패딩
        if len(audio_chunks[-1]) < audio_sample_per_frame:
            pad_length = audio_sample_per_frame - len(audio_chunks[-1])
            audio_chunks[-1] = np.pad(audio_chunks[-1], (0, pad_length), mode='constant')

        audio = np.array(audio_chunks, dtype=np.float32)  # (T, samples_per_frame)

        # Convert frames to PyTorch tensor (T, C, H, W)
        img_frames = frames_np[:, :, :, [2, 1, 0]]  # Convert BGR → RGB
        img_frames = torch.from_numpy(np.ascontiguousarray(np.transpose(img_frames, (0, 3, 1, 2)))).float()  # (T, C, H, W)

        # Resize to GT_size
        img_frames = F.interpolate(img_frames, size=GT_size, mode='nearest', align_corners=None)

        return {'Visual': img_frames, 'Audio': audio, 'Video_path' : path}

    def __len__(self):
        return len(self.dir_list)
