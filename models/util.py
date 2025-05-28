import torch
import torch.nn.functional as FU

def stft(data, n_fft = 255, hop_length = 500):
    window = torch.hann_window(n_fft).cuda()
    tmp = torch.stft(data, n_fft, hop_length, window = window, return_complex=True)
    tmp = torch.view_as_real(tmp)

    return tmp

def istft(data, n_fft = 255, hop_length = 500):
    window = torch.hann_window(n_fft).cuda()

    return torch.istft(torch.view_as_complex(data), n_fft, hop_length = hop_length, window = window, return_complex=False)
    

def stft_to_128x128_batch(signals: torch.Tensor,
                          n_fft: int = 510,
                          hop_length: int = 128,
                          sr: int = 16000) -> torch.Tensor:
    """
    입력: signals (B, T)  -- 배치 크기 B, 샘플 길이 T
    출력: D_128 (B, 128, 128, 32) -- 잘라낸 복합 STFT
    
    1) 입력 오디오 길이를 `511 * 255`로 패딩
    2) STFT 적용 후 shape (B, freq=513, time-=512, 2) 생성
    3) freq에서 맨 위 bin(1개) 제거 → (B, 512, 512, 2)
    4) (512,512,2) → (128,128,32)로 변환

    640 -> 256, 8  128 * 7 = 896
    """
    # 1) 모든 오디오 길이를 `511 * 255`로 맞춤
    desired_len = 896
    B, T = signals.shape
    
    if T < desired_len:
        pad_len = desired_len - T
        signals = FU.pad(signals, (0, pad_len))
        
    
    # 2) STFT 적용 => (B, freq=513, time=512, 2)
    window = torch.hann_window(n_fft, device=signals.device)
    D = torch.stft(signals,
                   n_fft=n_fft,
                   hop_length=hop_length,
                   window=window,
                   return_complex=True)
    
    # 복소수 -> (B, freq, time, 2) (real, imag)
    D = torch.view_as_real(D).contiguous()  # (B, 256, 128, 2)
    B, H, W, C = D.shape

    # 4) (256, 4, 2) → (64, 64, 1) 변환
    D = D.view(B, H // 4, W * 8, C//2).permute(0, 3, 1, 2).contiguous()
    return D


def istft_from_128x128_batch(D_128: torch.Tensor,
                             n_fft: int = 510,
                             hop_length: int = 128,
                             sr: int = 16000) -> torch.Tensor:
    """
    입력: D_128 (B, 128, 128, 32) -- 잘라낸 복합 STFT
    출력: signals_rec (B, 원래 T 길이) -- 복원된 오디오
    
    1) (128,128,32) → (512,512,2) 복원.
    2) 주파수 bin 복원 (0 추가하여 513으로 확장).
    3) iSTFT 수행 후 원래 길이로 복원.
    """
    B, C, H, W = D_128.shape  # (B, 128, 128, 32)

    # 1) (128,128,32) → (512,512,2) 복원
    D = D_128.permute(0, 2, 3, 1).contiguous().view(B, H * 4, W // 8, C * 2)

    # 3) iSTFT 수행
    window = torch.hann_window(n_fft, device=D.device)
    signals_rec = torch.istft(torch.view_as_complex(D),
                              n_fft=n_fft,
                              hop_length=hop_length,
                              window=window,
                              return_complex=False)

    return signals_rec[:, :640]  # 원래 길이로 잘라줌



import torch

def quantize_feature_map(feature_map):
    """
    입력된 피처 맵 텐서를 8비트(qint8)로 양자화합니다.

    Args:
        feature_map (torch.Tensor): 양자화할 피처 맵 텐서 (float32 타입)

    Returns:
        torch.Tensor: 양자화된 피처 맵 텐서 (qint8 타입)
    """
    # 스케일과 제로 포인트 계산 (대칭 양자화 사용)
    scale = feature_map.abs().max() / 127
    zero_point = 0  # 대칭 양자화이므로 0으로 설정

    # 텐서 양자화
    q_feature_map = torch.quantize_per_tensor(feature_map, scale, zero_point, dtype=torch.qint8)

    return q_feature_map
