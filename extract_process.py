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

def save_audio_wav(audio_tensor: torch.Tensor, out_path: str, sample_rate: int = 16000):
    """
    1~2채널 오디오 텐서를 .wav(PCM 16비트)로 저장
    
    Args:
        audio_tensor: (samples,) 모노 또는 (samples, channels) 형태
                      - 값 범위 [-1,1] 권장 -> 16bit PCM 변환 시 clipping
        out_path: 저장할 wav 파일 경로
        sample_rate: 오디오 샘플레이트
    """
    # 모노 -> (samples, 1) 변환
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(1)

    # [-1,1] 범위로 clamp 후 int16 변환
    audio_np = audio_tensor.clamp(-1, 1).cpu().numpy()  # float32/64
    audio_int16 = (audio_np * 32767.0).astype(np.int16)

    num_samples, num_channels = audio_int16.shape

    with wave.open(out_path, 'wb') as wf:
        wf.setnchannels(num_channels)  # 채널 수
        wf.setsampwidth(2)            # 샘플폭=16bit(2byte)
        wf.setframerate(sample_rate)   # 샘플레이트
        wf.writeframes(audio_int16.tobytes())

    print(f"[Wave] 오디오 저장 완료: {out_path}")


def save_video_opencv(video_tensor: torch.Tensor, out_path: str, fps: float = 25.0):
    """
    OpenCV로 영상(mp4)을 저장하는 함수 (오디오 X)
    
    Args:
        video_tensor: (T, C, H, W) 형태의 텐서
                      - 값 범위가 [0,1]이면 0~255로 스케일링
                      - 채널 순서가 BGR이면 그대로 저장됨
                        (RGB라면 BGR로 뒤집는 과정을 추가해야 색이 맞습니다)
        out_path: 결과로 저장할 mp4 파일 경로
        fps: 비디오의 프레임레이트
    """

    T, C, H, W = video_tensor.shape

    # (0~1 범위 가정 시) 0~255로 스케일링 후 uint8 변환
    video_np = (video_tensor.clamp(0,1) * 255).byte().cpu().numpy()  # (T, C, H, W)

    # (T, C, H, W) -> (T, H, W, C)
    video_np = np.transpose(video_np, (0, 2, 3, 1))  # OpenCV는 (H, W, C) 프레임

    # mp4v, avc1, XVID 등 가능. 여기서는 mp4v 사용
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    for i in range(T):
        frame = video_np[i]
        # video_tensor가 BGR이라면 그대로 사용 가능
        # 만약 RGB였다면 미리 video_tensor에서 채널을 뒤집었어야 함
        writer.write(frame)

    writer.release()
    print(f"[OpenCV] 비디오 저장 완료: {out_path}")


def combine_video_audio_ffmpeg(
    video_path: str, audio_path: str, out_path: str, audio_sr: int = 16000
):
    """
    FFmpeg 명령어로 (영상 + 오디오) 합쳐 하나의 mp4 생성
    
    Args:
        video_path: 영상만 있는 mp4 파일 경로
        audio_path: 오디오 wav 파일 경로
        out_path: 최종 출력 mp4 경로
        audio_sr: 오디오 샘플레이트 (ffmpeg에서 -ar로 지정)
    """
    cmd = [
        "ffmpeg", "-y",         # -y: 기존 파일 덮어쓰기
        "-i", video_path,       # 비디오 입력
        "-i", audio_path,       # 오디오 입력
        "-c:v", "copy",         # 비디오는 재인코딩 안 함
        "-c:a", "aac",          # 오디오는 aac 인코딩
        "-ar", str(audio_sr),   # 샘플레이트
        out_path
    ]
    print(f"[FFmpeg] 명령어: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[FFmpeg] 합성 완료: {out_path}")


def opencv_ffmpeg_save(
    video_tensor: torch.Tensor,
    audio_tensor: torch.Tensor,
    out_path: str,
    fps: float = 25.0,
    audio_sr: int = 16000
):
    """
    1) OpenCV로 영상(mp4) 저장
    2) 파이썬 wave로 오디오(wav) 저장
    3) FFmpeg로 둘을 합쳐 최종 out_path(mp4) 생성
    4) 임시 파일 제거

    Args:
        video_tensor: (T, C, H, W), [0,1] 범위, BGR이라면 그대로 사용 가능
        audio_tensor: (samples,) or (samples, channels), [-1,1] 범위 권장
        out_path: 최종 mp4 파일 경로
        fps: 영상 프레임레이트
        audio_sr: 오디오 샘플레이트
    """

    # 0) 임시 파일 경로 생성
    temp_video = f"temp_{uuid.uuid4().hex}.mp4"
    temp_audio = f"temp_{uuid.uuid4().hex}.wav"

    try:
        # 1) 비디오 저장
        save_video_opencv(video_tensor, temp_video, fps)

        # 2) 오디오 저장
        save_audio_wav(audio_tensor, temp_audio, sample_rate=audio_sr)

        # 3) FFmpeg로 합성
        combine_video_audio_ffmpeg(temp_video, temp_audio, out_path, audio_sr)

    finally:
        # 4) 임시 파일 삭제
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
        if phase == 'extract':

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
        avg_auc = 0.
        avg_ap = 0.
        avg_iou = [0. for _ in range(10)]
        avg_snr = 0.

        for video_id, val_data in enumerate(val_loader):
            img_dir = '/watermark/Interspeech_Test/STFT_DWT_FACIAL_SHORT_RESHAPE/save'
            video_path = val_data['Video_path']
            
            human_folder_name = video_path[0].split('/')[-2]
            file_name = video_path[0].split('/')[-1]
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
            k, auc, ap, iou, snr, pesq, extracted_audio, _ = model.extracting_test()
            if k == 0:
                continue
            total_idx += 1
            avg_pesq += pesq
            avg_snr += snr
            avg_auc += auc
            avg_ap += ap
            for i in range(10):
                avg_iou[i] += iou[i]

            pred_audio = util.tensor2audio(extracted_audio)

            audio_output_path = os.path.join(img_dir, 'EXT', human_folder_name, file_name.replace('.mp4', '.wav'))

            save_audio_path = os.path.join(img_dir,'EXT', '{:d}_{:s}.wav'.format(video_id, 'pred_audio'))
            util.save_audio(pred_audio, audio_output_path, args.fps)

        print({"SNR" : avg_snr/total_idx, "PESQ" : avg_pesq/total_idx, "AUC" : avg_auc/total_idx, "AP" : avg_ap/total_idx, "IOU" : [i/total_idx for i in avg_iou]})

if __name__ == '__main__':
    main()
