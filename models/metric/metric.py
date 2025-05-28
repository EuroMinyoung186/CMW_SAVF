import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from scipy.interpolate import interp1d

from sklearn.metrics import roc_auc_score
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim

################################################################################
# Audio Quality Metrics
################################################################################

def calculate_snr(clean_signal, noise_signal):
    signal_power = torch.mean(clean_signal ** 2)
    noise_power = torch.mean((clean_signal - noise_signal) ** 2)
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr

def calculate_stoi(clean_signal, noisy_signal, sampling_rate = 16000):
    return stoi(clean_signal, noisy_signal, sampling_rate, extended = False)

def calculate_pesq(ref_signal, deg_signal, sampling_rate = 16000):
    ref_signal = ref_signal.cpu().detach().numpy()
    deg_signal = deg_signal.cpu().detach().numpy()
    return pesq(sampling_rate, ref_signal, deg_signal, 'wb')

################################################################################
# Visual Quality Metrics
################################################################################    

def calculate_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    ssim_index, _ = ssim(img1, img2, full=True, multichannel=True, data_range=1.0,  win_size=3)
    return ssim_index

################################################################################
# Tamper Localization Metrics
################################################################################      

def compute_AUC(tensor, mask):
    tensor = 1 - tensor

    try:
        # roc_auc_score 함수를 사용하여 AUC 계산
        auc_t = roc_auc_score(mask.cpu().numpy(), tensor.cpu().detach().numpy())
    except ValueError as e:
        # 모든 레이블이 같거나 샘플 수가 충분하지 않을 때 예외 처리
        print(f"Error calculating AUC for timestep {t}: {e}")
        

    return auc_t

def compute_average_precision(sorted_labels):
    if not isinstance(sorted_labels, torch.Tensor):
        sorted_labels = torch.tensor(sorted_labels)

    sorted_labels = sorted_labels.float()
    total_positives = sorted_labels.sum()
    if total_positives == 0:
        return 0.0

    cumulative_relevance = sorted_labels.cumsum(dim=0)
    k_indices = torch.arange(1, len(sorted_labels) + 1, device=sorted_labels.device)
    precision_at_k = cumulative_relevance / k_indices
    precision_at_k = precision_at_k * sorted_labels
    ap = precision_at_k.sum() / total_positives
    return ap.item()


def compute_AP(similarity_tensor, binary_mask):
    sorted_similarities, sorted_indices = torch.sort(similarity_tensor, descending=False)
    sorted_labels = torch.gather(binary_mask, 0, sorted_indices)
    ap = compute_average_precision(sorted_labels)
    return ap

def compute_IoU(similarity_tensor, binary_mask, thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]){
    total_iou = [0] * len(thresholds)
    for idx, threshold in enumerate(thresholds):
        result = (iou_similarity <= threshold).float()

        intersection = (result * self.mask).sum(dim=0)
        union = ((result + self.mask) > 0).float().sum(dim=0)

        iou = intersection / union

        sum_iou = iou.sum()
        total_iou[idx] = sum_iou
}