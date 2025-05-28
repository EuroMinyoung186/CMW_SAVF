'''
Vimeo90K dataset
support reading images from lmdb, image folder and memcached
'''
import logging
import os
import os.path as osp
import pickle
import random
import librosa
import soundfile
import resampy

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import data.util as util

try:
    import mc  # import memcached
except ImportError:
    pass
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
        # get train indexes
        self.txt_path = self.opt['txt_path']
        self.list_video = []
        self.list_audio = [] 
        self.fps = self.opt['fps']
        

        with open(self.txt_path, 'r') as f:
            self.dir_list = f.readlines()
            self.dir_list = [a.strip('\n') for a in self.dir_list]

        for a in self.dir_list:
            audio_path = a.replace('frames', '16khz')
            files = [file for file in sorted(os.listdir(a)) if 'jpg' in file]
            self.list_video.append(files)
            self.list_audio.append(os.path.join(audio_path, 'audio.wav'))


        self.data_type = self.opt['data_type']
        self.audio_sample_rate = self.opt['audio_sample_rate']
        self.audio_sample = self.audio_sample_rate // self.fps

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def __getitem__(self, index):
        GT_size = self.opt['GT_size']
        video_paths = self.list_video[index]
        audio_path = self.list_audio[index]

        #tmp = random.randint(1, 7)
        #video_path = video_path + f'/im{str(tmp)}.png'
        frames = []
        for video_path in video_paths:
            frames.append(util.read_img(None, video_path))

        file_extension = os.path.splitext(audio_path)[-1].lower()

        if file_extension == '.mp3' or file_extension == '.wav':
            audio, original_sr = librosa.load(audio_path, sr=None)

        if len(audio.shape) == 2:
            audio = audio[:, 0]

        if original_sr != self.audio_sample_rate:
            audio = resampy.resample(audio, original_sr, self.audio_sample_rate)
        audio = [audio[i*self.audio_sample:(i+1) * self.audio_sample] for i in range(len(frames))]

        current_length = len(audio[-1])
        if current_length < self.audio_sample:
            pad_length = self.audio_sample - current_length
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
        audio = np.array(audio)
        
        # shape : T H W C
        img_frames = np.stack(frames, axis=0)

        # shape : T C H W
        img_frames = img_frames[:, :, :, [2, 1, 0]] # BGR to RGB
        img_frames = torch.from_numpy(np.ascontiguousarray(np.transpose(img_frames, (0, 3, 1, 2)))).float() # T C H W
                
        # random crop
        img_frames = torch.nn.functional.interpolate(img_frames, size=GT_size, mode='nearest', align_corners=None)

        return {'Visual': img_frames, 'Audio': audio.astype(np.float32)}

    def __len__(self):
        assert len(self.list_video) <= len(self.list_audio)
        return len(self.list_video)