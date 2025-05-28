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
        self.base_path = self.opt['base_path']
        self.version = self.opt['version']

        self.audio_sample_rate = self.opt['audio_sample_rate']
        self.audio_sample = self.audio_sample_rate // self.fps

        mask_paths = os.path.join(self.base_path, 'VC_MASK')
        video_paths = os.path.join(self.base_path, 'Wav2Lip')
        audio_paths = os.path.join(self.base_path, 'AUD')
        attack_paths = os.path.join(self.base_path, 'VC_EXCHANGE')

        self.list_video = []
        self.list_audio = []
        self.list_mask = []
        self.list_attack = []
        print(video_paths)
        for base_name in os.listdir(video_paths):
            video_path = os.path.join(video_paths, base_name)
            audio_path = os.path.join(audio_paths, base_name)
            mask_path = os.path.join(mask_paths, base_name)
            attack_path = os.path.join(attack_paths, base_name)
            
            for video, audio, mask, attack in zip(sorted(os.listdir(video_path), reverse = True), sorted(os.listdir(audio_path),  reverse = True), sorted(os.listdir(mask_path),  reverse = True), sorted(os.listdir(attack_path),  reverse = True)):
                self.list_video.append(os.path.join(video_path, video))
                self.list_audio.append(os.path.join(audio_path, audio))
                self.list_mask.append(os.path.join(mask_path, mask))
                self.list_attack.append(os.path.join(attack_path, attack))
                

    def __getitem__(self, index):
        GT_size = self.opt['GT_size']
        video_path = self.list_video[index]
        audio_path = self.list_audio[index]
        mask_path = self.list_mask[index]
        attack_path = self.list_attack[index]
        

        # Load video clip
        clip = VideoFileClip(video_path)
        duration = clip.duration
        n_frames = int(duration * self.fps)
        
        times = np.linspace(0, duration, n_frames, endpoint=False)

        # Extract frames
        frames = [clip.get_frame(t) for t in times]
        frames_np = np.array(frames)
        
        masks = np.load(mask_path)
        print(audio_path)
        print(video_path)

        # Extract audio and resample to 4000 Hz
        #audio = clip.audio.to_soundarray(fps=clip.audio.fps)  # 원본 샘플링 레이트로 가져옴
        #orig_sr = clip.audio.fps  # 원본 샘플링 레이트

        #if len(audio.shape) == 2:  # Stereo → Mono 변환
        #    audio = np.mean(audio, axis=1)

        # Resample to 4000 Hz
        #if orig_sr != self.audio_sample_rate:
        #    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.audio_sample_rate)
        #print(len(audio))

        # Slice audio to match frame count
        audio_sample_per_frame = self.audio_sample_rate // self.fps
        #audio_chunks = [
        #    audio[i * audio_sample_per_frame:(i + 1) * audio_sample_per_frame]
        #    for i in range(n_frames)
        #]
        #print(audio_sample_per_frame)

        # 마지막 청크가 충분한 길이가 아니라면 패딩
        #if len(audio_chunks[-1]) < audio_sample_per_frame:
        #    pad_length = audio_sample_per_frame - len(audio_chunks[-1])
        #    audio_chunks[-1] = np.pad(audio_chunks[-1], (0, pad_length), mode='constant')

        #audio = np.array(audio_chunks, dtype=np.float32)  # (T, samples_per_frame)

        # Convert frames to PyTorch tensor (T, C, H, W)
        img_frames = frames_np[:, :, :, [2, 1, 0]]  # Convert BGR → RGB
        img_frames = torch.from_numpy(np.ascontiguousarray(np.transpose(img_frames, (0, 3, 1, 2)))).float()  # (T, C, H, W)
        img_frames /= 255.0  # Normalize to [0,1]

        # Resize to GT_size
        img_frames = F.interpolate(img_frames, size=GT_size, mode='nearest', align_corners=None)

        img_frames = img_frames[:125]
        n_frames = 125

        orig_audio, sr = librosa.load(audio_path, sr=None)
        if len(orig_audio.shape) == 2:
            orig_audio = orig_audio[:, 0]

        

        

        
        

        attack_audio, sr = librosa.load(attack_path, sr=None)
        if len(attack_audio.shape) == 2:
            attack_audio = attack_audio[:, 0]

        orig_audio = orig_audio[:n_frames * audio_sample_per_frame]
        attack_audio = attack_audio[:n_frames * audio_sample_per_frame]


        attack_audio_chunk = [
            attack_audio[i * audio_sample_per_frame:(i + 1) * audio_sample_per_frame] 
            for i in range(n_frames)
        ]

        # 마지막 청크가 충분한 길이가 아니라면 패딩
        #if len(attack_audio_chunk[-1]) < audio_sample_per_frame:
        #    pad_length = audio_sample_per_frame - len(attack_audio_chunk[-1])
        #    attack_audio_chunk[-1] = np.pad(attack_audio_chunk[-1], (0, pad_length), mode='constant')

        for i in range(len(attack_audio_chunk)):
            if len(attack_audio_chunk[i]) < audio_sample_per_frame:
                pad_length = audio_sample_per_frame - len(attack_audio_chunk[i])
                attack_audio_chunk[i] = np.pad(attack_audio_chunk[i], (0, pad_length), mode='constant')

        attack_audio = np.array(attack_audio_chunk, dtype=np.float32)  # (T, samples_per_frame)

        return {'Visual': img_frames, 'Audio': attack_audio, 'Video_path' : video_path, 'Mask' : masks, 'Orig_Audio' : orig_audio}

    def __len__(self):
        return len(self.list_video)
