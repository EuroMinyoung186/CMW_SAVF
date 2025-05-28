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

import data.util as util
from moviepy.editor import VideoFileClip

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
        self.video_path = self.opt['video_path']
        self.audio_path = self.opt['audio_path']
        self.video_mask_path = self.opt['video_mask_path']
        self.video_txt_path = self.opt['video_txt_path']
        self.audio_txt_path = self.opt['audio_txt_path']

        with open(self.video_txt_path) as f:
            self.video_list = f.readlines()
            
        with open(self.audio_txt_path) as f:
            self.audio_list = f.readlines()

        self.list_video = [line.strip('\n') for line in self.video_list]
        self.list_audio = [line.strip('\n') for line in self.audio_list]

        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        
        self.data_type = self.opt['data_type']
        self.audio_sample_rate = self.opt['audio_sample_rate']
        random.shuffle(self.list_video)
        self.LR_input = True

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def __getitem__(self, index):
        GT_size = self.opt['GT_size']
        video_name = self.list_video[index]
        audio_name = self.list_audio[index]

        path_frame = os.path.join(self.video_path, video_name)
        path_audio = os.path.join(self.audio_path, audio_name)
        path_mask_frame = os.path.join(self.video_mask_path, video_name)

        clip = VideoFileClip(path_frame)
        print(clip.fps)
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in clip.iter_frames()]

        file_extension = os.path.splitext(path_audio)[1].lower()

        if file_extension == '.mp3':
            audio, original_sr = librosa.load(path_audio, sr=None)
        elif file_extension == '.wav':
            audio, original_sr = soundfile.read(path_audio)

        if len(audio.shape) == 2:
            audio = audio[:, 0]
            

        if original_sr != self.audio_sample_rate:
            audio = resampy.resample(audio, original_sr, self.audio_sample_rate)
        
        audio_length_second = 1.0 * len(audio) / self.audio_sample_rate
                
        # random crop
        H, W, C = frames[0].shape
        rnd_h = random.randint(0, max(0, H - GT_size))
        rnd_w = random.randint(0, max(0, W - GT_size))
        frames = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in frames]

        # shape : T H W C
        img_frames = np.stack(frames, axis=0)

        # shape : T C H W
        img_frames = img_frames[:, :, :, [2, 1, 0]] # BGR to RGB
        img_frames = torch.from_numpy(np.ascontiguousarray(np.transpose(img_frames, (0, 3, 1, 2)))).float() # T C H W

        # process audio
        T, _ , _, _ = img_frames.size()
        samples_per_frame = int((audio_length_second * self.audio_sample_rate) // T)
  

        sliced_audio = [audio[i * samples_per_frame:(i + 1) * samples_per_frame] for i in range(T)]
        
        sliced_audio = np.stack(sliced_audio, axis=0)
        


        return {'Visual': img_frames, 'Audio': sliced_audio}

    def __len__(self):
        assert len(self.list_video) == len(self.list_audio)
        return len(self.list_video)