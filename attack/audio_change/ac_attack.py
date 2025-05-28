import os
import random
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
from imageio import imwrite
import librosa
import soundfile
from tqdm import tqdm

# 기본 경로 설정
file_path = '/watermark/HDTF_dataset/test.txt'  # test.txt에는 처리할 폴더들의 경로가 한 줄에 하나씩 들어있다고 가정
base_path = '/watermark/Interspeech_Test/STFT_DWT_FACIAL/save/AUD'
output_path = '/watermark/Interspeech_Test/STFT_DWT_FACIAL/save/EXCHANGE'

os.makedirs(output_path, exist_ok=True)
random.seed(42)
idx = 0
# test.txt 파일에서 경로들을 읽어옵니다.
with open(file_path, 'r') as f:
    filepaths = f.readlines()
    filepaths = [filepath.strip() for filepath in filepaths if filepath.strip()]

def derangement(lst):
    """
    입력 리스트의 순서를 무작위로 섞되,
    어떤 원소도 자기 자신과 대응되지 않는 (즉, 제자리 치환이 없는) 순열을 반환합니다.
    """
    while True:
        perm = random.sample(lst, len(lst))
        if all(a != b for a, b in zip(lst, perm)):
            return perm

# --- 각 폴더(파일 경로)별 처리 ---
for filepath in filepaths:
    # 예: filepath = '/watermark/HDTF_dataset/test/RD_Radio34_002'
    last_name = os.path.basename(filepath)
    os.makedirs(os.path.join(output_path, last_name), exist_ok=True)
    replace_path = os.path.join(base_path, last_name)

    # 해당 폴더 내의 mp4 파일들을 정렬하여 리스트 생성
    wavfile = sorted(os.listdir(replace_path))
    wavfile = [os.path.join(replace_path, f)  for f in wavfile if f.endswith('.wav')]

    if not wavfile:
        print(f"{replace_path} 에 mp4 파일이 없습니다.")
        continue

    # (참고: 오디오 교체에 사용할 대체 파일은 현재 처리 중인 파일를 제외한 파일들에서 임의 선택합니다.)
    # 아래 possible_files 는 각 파일 처리 시마다 사용됩니다.

    print(f"\n--- 폴더 {filepath} 처리 시작 ---")
    # 각 mp4 파일 처리
    target_sr = 16000 
    for file in tqdm(wavfile):
        audio, sr = librosa.load(file, sr=16000)
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        num_samples = audio.shape[0]
        new_audio = audio.copy()
        mask = np.zeros(num_samples, dtype=np.int32)
        
        seg1_length = int(0.10 * num_samples)
        seg2_length = int(0.30 * num_samples)
        
        start1 = random.randint(0, num_samples - seg1_length)
        start2 = random.randint(0, num_samples - seg2_length)

        print('진입')

        while not (start2 + seg2_length <= start1 or start2 >= start1 + seg1_length):
            start2 = random.randint(0, num_samples - seg2_length)
        
        print(f"  seg1 (10%): 시작 {start1}, 길이 {seg1_length} samples")
        print(f"  seg2 (50%): 시작 {start2}, 길이 {seg2_length} samples")
        
        # possible_files: 현재 파일 제외한 mp4_paths에서 선택
        possible_files = [f for f in wavfile if f != file]
        replacement_file1 = random.choice(possible_files)
        rep_audio1, sr = librosa.load(replacement_file1, sr=16000)
        rep_num_samples1 = rep_audio1.shape[0]
        if rep_num_samples1 < seg1_length:
            start_rep1 = 0
            seg1_length = rep_num_samples1  # 대체 구간 길이 조정
        else:
            start_rep1 = random.randint(0, rep_num_samples1 - seg1_length)

        rep_segment1 = rep_audio1[start_rep1:start_rep1 + seg1_length]

        replacement_file2 = random.choice(possible_files)
        rep_audio2, sr = librosa.load(replacement_file2, sr=16000)
        rep_num_samples2 = rep_audio2.shape[0]
        if rep_num_samples2 < seg2_length:
            start_rep2 = 0
            seg2_length = rep_num_samples2
        else:
            start_rep2 = random.randint(0, rep_num_samples2 - seg2_length)
        rep_segment2 = rep_audio2[start_rep2:start_rep2 + seg2_length]
        # 교체: new_audio의 seg1, seg2 구간을 대체 구간으로 변경
        new_audio[start1:start1+seg1_length] = rep_segment1
        new_audio[start2:start2+seg2_length] = rep_segment2
        
        # mask: 교체한 구간에 대해 1로 표시
        mask[start1:start1+seg1_length] = 1
        mask[start2:start2+seg2_length] = 1

        # 출력 폴더 생성 (full_change 폴더 아래, 파일명 기준)
        base_name = os.path.splitext(os.path.basename(file))[0]
        mask_path = replace_path.replace('/AUD', '/MASK')
        os.makedirs(mask_path, exist_ok=True)
        audio_output = os.path.join(replace_path, f"{base_name}.wav")
        mask_output = os.path.join(mask_path, f"{base_name}.npy")
        
        print(f"  새 오디오 저장: {audio_output}")
        soundfile.write(audio_output.replace('/AUD', '/EXCHANGE'), new_audio, sr, format='WAV')

        np.save(mask_output, mask)
        print(f"  mask 저장: {mask_output} (바뀐 부분은 1, 길이: {mask.shape[0]} samples)")
        idx += 1
        
        
    print("\n모든 파일 처리 완료!")
