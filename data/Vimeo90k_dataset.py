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
import face_recognition

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

import data.util as util

try:
    import mc  # import memcached
except ImportError:
    pass
logger = logging.getLogger('base')

class Vimeo90KDataset(data.Dataset):
    '''
    Reading the training Vimeo90K dataset
    key example: 00001_0001 (_1, ..., _7)
    GT (Ground-Truth): 4th frame;
    LQ (Low-Quality): support reading N LQ frames, N = 1, 3, 5, 7 centered with 4th frame
    '''

    def __init__(self, opt):
        super(Vimeo90KDataset, self).__init__()
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
            
            for idx in range(0, len(files)):
    
                self.list_video.append(os.path.join(a, files[idx]))
                self.list_audio.append((os.path.join(audio_path, 'audio.wav'), idx))

        self.random_reverse = opt['random_reverse']
        #if self.random_reverse:
        #    random.shuffle(self.list_video)

        self.data_type = self.opt['data_type']
        self.audio_sample_rate = self.opt['audio_sample_rate']

        self.audio_length = self.opt['audio_length']
    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def __getitem__(self, index):
        GT_size = self.opt['GT_size']
        video_path = self.list_video[index]
        audio_path, idx = self.list_audio[index]

        #tmp = random.randint(1, 7)
        #video_path = video_path + f'/im{str(tmp)}.png'
        frames = []
        
        frames.append(util.read_img(None, video_path))
        for_mask_frame = frames[0]

        if for_mask_frame.dtype == np.float32 or for_mask_frame.dtype == np.float64:
            for_mask_frame = (for_mask_frame * 255).astype(np.uint8)

        for_mask_frame = cv2.cvtColor(for_mask_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(for_mask_frame)

        mask = np.zeros((GT_size, GT_size), dtype=np.uint8)


        for (top, right, bottom, left) in face_locations:
            mask[max(0, top-10):bottom+10, max(0, left-10):right+10] = 1.


        file_extension = os.path.splitext(audio_path)[-1].lower()

        if file_extension == '.mp3' or file_extension == '.wav':
            audio, original_sr = librosa.load(audio_path, sr=None)

        if len(audio.shape) == 2:
            audio = audio[:, 0]

        if original_sr != self.audio_sample_rate:
            audio = resampy.resample(audio, original_sr, self.audio_sample_rate)

        start = random.randint(0, len(audio) - self.audio_length * self.audio_sample_rate)
        audio = audio[start : start + self.audio_length * self.audio_sample_rate]
        
                
        # random crop
        H, W, C = frames[0].shape
        #rnd_h = random.randint(0, max(0, H - GT_size))
        #rnd_w = random.randint(0, max(0, W - GT_size))
        #frames = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in frames]

        # shape : T H W C
        img_frames = np.stack(frames, axis=0)

        # shape : T C H W
        img_frames = img_frames[:, :, :, [2, 1, 0]] # BGR to RGB
        img_frames = torch.from_numpy(np.ascontiguousarray(np.transpose(img_frames, (0, 3, 1, 2)))).float() # T C H W

        return {'Visual': img_frames, 'Audio': audio.astype(np.float32), 'Mask' : mask.astype(np.float32)}

    def __len__(self):
        assert len(self.list_video) == len(self.list_audio)
        return len(self.list_video)