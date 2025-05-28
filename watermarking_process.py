import os
import math
import uuid
import argparse
import random
import logging
import warnings

import cv2
import ffmpeg
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from torch.backends import cudnn
import torchvision

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from models.metric.metric import calculate_snr, calculate_psnr, calculate_ssim, compute_AP_metric_cos
import moviepy.editor as mpy
from moviepy.audio.AudioClip import AudioArrayClip 

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def cal_pnsr(sr_img, gt_img):
    # calculate PSNR
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.
    psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)

    return psnr

import subprocess
import os

import cv2
import torch
import numpy as np

import wave

def save_video_ffmpeg(video_tensor: torch.Tensor, out_path: str, fps: float = 25.0, codec: str = "libx264", lossless: bool = True):
    """
    FFmpeg로 무손실 영상(mp4)을 저장하는 함수
    
    Args:
        video_tensor: (T, C, H, W) 형태의 텐서 (값 범위 [0,1]이면 0~255로 스케일링)
        out_path: 결과로 저장할 mp4 파일 경로
        fps: 비디오의 프레임레이트
        codec: 압축 코덱 ("libx264"=H.264, "libx265"=H.265)
        lossless: 무손실 설정 여부 (True이면 완전 무손실 압축 적용)
    """
    import os, uuid, subprocess, numpy as np
    import cv2
    import torch

    T, C, H, W = video_tensor.shape
    video_np = (video_tensor.clamp(0,1) * 255).byte().cpu().numpy()  # (T, C, H, W)
    video_np = np.transpose(video_np, (0, 2, 3, 1))  # (T, H, W, C)

    # 임시 이미지 프레임 저장 디렉토리
    temp_dir = f"temp_{uuid.uuid4().hex}"
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(T):
        frame_path = os.path.join(temp_dir, f"{i:04d}.png")
        cv2.imwrite(frame_path, video_np[i])  # PNG는 무손실 저장

    # 시스템 FFmpeg 전체 경로로 명시
    ffmpeg_path = "/usr/bin/ffmpeg"  # 또는 /usr/local/bin/ffmpeg

    ffmpeg_cmd = [
        ffmpeg_path, "-y",
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "%04d.png"),
        "-c:v", codec,
        "-r", str(fps)
    ]

    if lossless:
        if codec == "libx264":
            ffmpeg_cmd += ["-crf", "0", "-preset", "veryslow", "-pix_fmt", "yuv444p"]
        elif codec == "libx265":
            ffmpeg_cmd += ["-x265-params", "lossless=1", "-pix_fmt", "yuv444p"]

    ffmpeg_cmd.append(out_path)

    print(f"[FFmpeg] 영상 저장 명령어: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True)

    # 임시 이미지 삭제
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    print(f"[FFmpeg] 비디오 저장 완료: {out_path}")


def save_audio_wav(audio_tensor: torch.Tensor, out_path: str, sample_rate: int = 16000):
    """
    오디오 텐서를 무손실 WAV 파일로 저장하는 함수
    """
    audio_np = audio_tensor.cpu().numpy()
    audio_np = (audio_np * 32767).astype(np.int16)  # 16-bit PCM 변환

    with wave.open(out_path, "w") as wav_file:
        num_channels = 1 if len(audio_np.shape) == 1 else audio_np.shape[1]
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_np.tobytes())

    print(f"[WAV] 오디오 저장 완료: {out_path}")

def combine_video_audio_ffmpeg(video_path: str, audio_path: str, out_path: str, audio_sr: int = 16000):
    """
    FFmpeg 명령어로 (영상 + 오디오) 합쳐 하나의 mp4 생성
    MP4 컨테이너에서는 FLAC 지원이 안 되므로 AAC로 변경.
    
    Args:
        video_path: 영상만 있는 mp4 파일 경로
        audio_path: 오디오 wav 파일 경로
        out_path: 최종 출력 mp4 경로
        audio_sr: 오디오 샘플레이트 (ffmpeg에서 -ar로 지정)
    """

    # MP4는 FLAC 지원 안 되므로 AAC로 변경
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",  # 비디오는 그대로 유지
        "-c:a", "aac",  # AAC 코덱 사용 (MP4 호환)
        "-b:a", "256k",  # 오디오 품질 설정 (256 kbps)
        "-ar", str(audio_sr),  # 샘플레이트 설정
        out_path
    ]

    print(f"[FFmpeg] 합성 명령어: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[FFmpeg] 합성 완료: {out_path}")


def opencv_ffmpeg_save(video_tensor: torch.Tensor, audio_tensor: torch.Tensor, out_path: str, fps: float = 25.0, audio_sr: int = 16000, codec: str = "libx264", lossless: bool = True):
    """
    1) FFmpeg로 무손실 영상(mp4, H.264/H.265) 저장
    2) 파이썬 wave로 무손실 오디오(wav) 저장
    3) FFmpeg로 둘을 합쳐 최종 out_path(mp4) 생성
    4) 임시 파일 제거
    """

    temp_video = f"temp_{uuid.uuid4().hex}.mp4"
    temp_audio = f"temp_{uuid.uuid4().hex}.wav"

    try:
        save_video_ffmpeg(video_tensor, temp_video, fps, codec, lossless)
        save_audio_wav(audio_tensor, temp_audio, sample_rate=audio_sr)
        combine_video_audio_ffmpeg(temp_video, temp_audio, out_path, audio_sr)
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


def main():
    import wandb
    
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.', default="/watermark/Interspeech_Test/STFT_DWT_FACIAL_SHORT_RESHAPE/options/test/val_LF-VSN_1video.yml")  # config 文件
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--fps', type=int, default=16000)                  
    parser.add_argument('--project_name', type=str, default='full')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--remove', type=bool, default=False)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    opt['remove'] = args.remove
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0  # Default to 0 if not set

    # Set the device
    torch.cuda.set_device(args.local_rank)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    opt['name'] = args.project_name
    # distributed training settings
    if args.launcher == 'pytorch':
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    opt['rank'] = rank

    if rank == 0:

        wandb.init(project=args.project_name, config={
            "learning_rate": 0.001,
            "dropout": 0.2,
            "architecture": "LF-VSN",
            "dataset": "Vimeo90k",
        })
        config = wandb.config

    else:
        wandb.init(mode="disabled")

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True
    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':

            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
    
    # create model
    model = create_model(opt)
    # resume training
    if resume_state:

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    

    # validation
    if rank <= 0:
        total_idx = 0.0
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_pesq = 0.
        avg_snr = 0.

        for video_id, val_data in enumerate(val_loader):
            img_dir = '/watermark/Interspeech_Test/STFT_DWT_FACIAL_SHORT_RESHAPE/save'
            video_path = val_data['Video_path']
            
            human_folder_name = video_path[0].split('/')[-1]
            file_name = 'video.mp4'
            util.mkdir(img_dir)

            util.mkdir(os.path.join(img_dir, 'GT'))
            util.mkdir(os.path.join(img_dir, 'AUD'))
            util.mkdir(os.path.join(img_dir, 'EXT'))

            util.mkdir(os.path.join(img_dir, 'CON'))
            util.mkdir(os.path.join(img_dir, 'Video'))
            os.makedirs(os.path.join(img_dir, 'Video', human_folder_name), exist_ok = True)
            os.makedirs(os.path.join(img_dir, 'AUD', human_folder_name), exist_ok = True)
            os.makedirs(os.path.join(img_dir, 'EXT', human_folder_name), exist_ok = True)

            output_path = os.path.join(img_dir, 'Video', human_folder_name, file_name)
            
            
            model.feed_data(val_data)
            T, psnr, ssim, pesq, snr, container, audio, pred_audio, GT = model.test_without_mask()
            
            total_idx += T
            avg_psnr += psnr
            avg_ssim += ssim
            avg_pesq += pesq
            avg_snr += snr

            aud = util.tensor2audio(audio)
            pred_audio = util.tensor2audio(pred_audio)

            audio_output_path = os.path.join(img_dir, 'EXT', human_folder_name, file_name.replace('.mp4', '.wav'))

            save_audio_path = os.path.join(img_dir,'EXT', '{:d}_{:s}.wav'.format(video_id, 'pred_audio'))
            util.save_audio(pred_audio, audio_output_path, args.fps)

            audio_output_path = os.path.join(img_dir, 'AUD', human_folder_name, file_name.replace('.mp4', '.wav'))

            save_audio_path = os.path.join(img_dir,'AUD', '{:d}_{:s}.wav'.format(video_id, 'audio'))
            util.save_audio(aud, audio_output_path,args.fps)

            #frames = container.permute(0, 2, 3, 1).contiguous()
            print(audio.shape)
            print(container.shape)

            save_video_ffmpeg(container, output_path)

            '''
            torchvision.io.write_video(
                output_path,
                video_array=container.permute(0, 2, 3, 1).cpu() ,fps = 25, video_codec='libx264',
                audio_array=audio.unsqueeze(0).cpu().clamp(-1, 1), audio_fps=16000, audio_codec='aac'
            ) 
            '''

            for i in range(T):
                
                img_con = util.tensor2img(container[i])
                img_gt = util.tensor2img(GT[i])

                
                save_img_path = os.path.join(img_dir, 'GT', '{:d}_{:d}_{:s}.png'.format(video_id, i, 'GT'))
                util.save_img(img_gt, save_img_path)

                save_img_path = os.path.join(img_dir, 'CON', '{:d}_{:d}_{:s}.png'.format(video_id, i, 'Con'))
                util.save_img(img_con, save_img_path)

        print({ "PSNR": avg_psnr/total_idx, "SSIM" : avg_ssim/total_idx, "SNR" : avg_snr/total_idx, "PESQ" : avg_pesq/total_idx})

if __name__ == '__main__':
    main()
