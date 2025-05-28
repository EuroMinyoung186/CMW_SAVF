import os
import math
import random
import logging
from tqdm import tqdm
from collections import OrderedDict

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.networks as networks
import models.lr_scheduler as lr_scheduler


from .base_model import BaseModel
from models.metric.metric import *
from models.network.builder import MoCo
from models.modules.common import DWT, IWT
from models.modules.Quantization import Quantization
from models.classifier.Mask_Generator import MixedMaskEmbedder
from models.modules.loss import ReconstructionLoss, ContrastiveLoss
from models.util import istft_from_128x128_batch, stft_to_128x128_batch

from noise.audio_noise import *
from noise.video_noise.JPEG import DiffJPEG

logger = logging.getLogger('base')
dwt = DWT()
iwt = IWT()

class Model_VSN(BaseModel):
    def __init__(self, opt):
        super(Model_VSN, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = -1  # non dist training
            self.world_size = 1

        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.opt_net = opt['network_G']
        self.m = opt['moco_m']
        self.total_epoch = opt['total_epochs']
        self.chunk_size = opt['chunk_size']
        self.mask_type = opt['mask_type']

        self.netG = networks.define_G_v2(opt).to(self.device)
        self.moco = MoCo().to(self.device)
        self.mask_embedder = MixedMaskEmbedder()
        self.audio_segment = opt['message_length']
        self.audio_noise = opt['audio_noise']

        self.lambda_vrl = opt['train']['lambda_vrl']
        self.lambda_wl = opt['train']['lambda_wl']
        self.lambda_arl = opt['train']['lambda_arl']
        self.lambda_sfcl = opt['train']['lambda_sfcl']
        
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            self.moco = DistributedDataParallel(self.moco, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
            self.moco = DataParallel(self.moco)

        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()
            self.moco.train()

            # loss
            self.Reconstruction_forw = MSELoss(reduction='sum')
            self.Reconstruction_back = MSELoss(reduction='sum')
            self.Reconstruction_center = MSELoss(reduction='sum')

            self.criterion = nn.CrossEntropyLoss()

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            wd_E = train_opt['weight_decay_E'] if train_opt['weight_decay_E'] else 0

            optim_params_G = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params_G.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params_G, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            
            self.optimizer_E = torch.optim.Adam(filter(lambda p: p.requires_grad, self.moco.module.parameters()), 
                                                lr=train_opt['lr_E'], weight_decay=wd_E, 
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_E)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
                    
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
                    
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.real_H = data['Visual'].to(self.device)
        self.ref_L = data['Audio'].to(self.device)
        if 'Mask' in data:
            self.mask = data['Mask'].to(self.device)
        if 'Orig_Audio' in data:
            self.orig_audio = data['Orig_Audio'].to(self.device)

    
    @torch.no_grad()
    def concat_all_gather(self, tensor):
        if self.world_size > 1:
            tensors_gather = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            torch.distributed.all_gather(tensors_gather, tensor)
            output = torch.cat(tensors_gather, dim=0)
            return output
        else:
            return tensor

    def optimize_parameters(self, epoch, wandb, train_type='inn'):
        self.netG.train()
        self.moco.train()

        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()

        self.adjust_learning_rate(self.optimizer_E, epoch, self.total_epoch, self.train_opt)

        hide_audio = self.ref_L[:, :self.audio_segment] 
        aud_b, aud_sample = hide_audio.shape

        cover_video = self.real_H.view(vis_b*vis_t, vis_c, vis_h, vis_w)
        vis_b, vis_t, vis_c, vis_h, vis_w = self.real_H.shape
        
        tmp = dwt(cover_video)
        tmp_b, tmp_c, tmp_h, tmp_w = tmp.shape
        tmp = tmp.view(tmp_b, tmp_c * 4, tmp_h // 2, tmp_w // 2) # Reshape it

        stft_audio = stft_to_128x128_batch(hide_audio)

        container, _ = self.netG(x=tmp, x_h=stft_audio, rev=False)

        container = container.view(vis_b*vis_t, -1, tmp_h, tmp_w)
        container = iwt(container).clamp(0, 1)

        Gt_ref = self.real_H.view(vis_b*vis_t, vis_c, vis_h, vis_w).detach()
        l_forw_fit = self.Reconstruction_forw(container, Gt_ref)
    
        choice = random.randint(0, 2)

        open_audio = hide_audio.view(aud_b, aud_sample)
        
        if choice == 0:
            NL = float((np.random.randint(1, 16))/255)
            noise = np.random.normal(0, NL, container.shape)
            torchnoise = torch.from_numpy(noise).cuda().float()
            container = container + torchnoise

        elif choice == 1:
            NL = int(np.random.randint(70,95))
            self.DiffJPEG = DiffJPEG(differentiable=True, quality=int(NL)).cuda()
            container = self.DiffJPEG(container)
            
        
        elif choice == 2:
            vals = 10**4

            container_non_neg = torch.clamp(container, min=0)
            if random.random() < 0.5:
                noisy_img_tensor = torch.poisson(container_non_neg * vals) / vals
            else:
                img_gray_tensor = torch.mean(container_non_neg, dim=0, keepdim=True)
                noisy_gray_tensor = torch.poisson(img_gray_tensor * vals) / vals
                noisy_img_tensor = container_non_neg + (noisy_gray_tensor - img_gray_tensor)

            container = torch.clamp(noisy_img_tensor, 0, 1)

        if self.mask_type == 'facial':
            dummy_img = np.zeros(container.shape)
            self.mask = self.mask_embedder(dummy_img, None, verbose=True).to(self.device)

        elif self.mask_type == 'random':
            self.mask = self.mask.unsqueeze(1)

        container = (1-self.mask) * container + self.mask * Gt_ref
        GT_mask = 1 - self.mask

        open_audio = open_audio.contiguous()
        noised_audio, T = self.divide_to_chunk(open_audio, self.chunk_size)
        noised_audio = self.moco.module.extract_features_q(noised_audio)   # B*32, 256

        tmp = dwt(container).contiguous()
        tmp = tmp.view(vis_b*vis_t, -1, tmp_h // 2, tmp_w // 2)
        
        out_x, out_x_h, out_z = self.netG(x=tmp, rev=True) #BT, 640
        out_x_h = out_x_h.contiguous()
       
        out_x_h = istft_from_128x128_batch(out_x_h).contiguous()
        l_center_x = self.Reconstruction_center(out_x_h.view(vis_b, -1), hide_audio.view(vis_b, -1).detach()) 

        out_x = out_x.contiguous()
        out_x = out_x.view(vis_b*vis_t, -1, tmp_h, tmp_w)
        out_x = iwt(out_x)

        host = self.real_H.detach()
        hide_audio = hide_audio.view(vis_b, -1)
        out_x_h = out_x_h.view(vis_b, -1)

        original_out_x_h, T = self.divide_to_chunk(out_x_h, self.chunk_size)
        original_out_x_h = self.moco.module.extract_features_k(original_out_x_h)

        l_back_rec = self.Reconstruction_back(out_x.view(vis_b, -1), host.view(vis_b, -1))

        logits, labels = self.moco.module.contrastive_loss(noised_audio, original_out_x_h)
        contrastive_loss = self.criterion(logits, labels)

        loss =  l_forw_fit * self.lambda_wl + l_center_x * self.lambda_vrl + l_back_rec * self.lambda_arl + contrastive_loss *self.lambda_sfcl 
        
        
        if self.world_size > 1:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss /= self.world_size
        
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        loss.backward()

        # set log
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_contrastive'] = contrastive_loss.item()
        self.log_dict['l_center_x'] = l_center_x.item()

        self.optimizer_E.step()
        self.optimizer_G.step()
            
    def extracting_test(self):
        self.netG.eval()
        self.moco.eval()

        torch.cuda.empty_cache()
        with torch.no_grad():
            self.extracted_audios = []
            self.changed_audios = []
            iou_similarities = []

            vis_b, vis_t, vis_c, vis_h, vis_w = self.real_H.shape
            aud_b, aud_t, aud_sample = self.ref_L.shape

            total_snr = 0.
            total_pesq = 0.
            total_stoi = 0.
            total_auc = 0.
            total_ap = 0.
            total_iou = [0. for i in range(10)]
            
            for t in tqdm(range(vis_t)):
                attack_audio = self.ref_L[:, t] # B, 640
                cover_video = self.real_H[:, t] # B, 3, 512, 512

                tmp = dwt(cover_video)
                tmp_b, tmp_c, tmp_h, tmp_w = tmp.shape

                tmp = tmp.view(tmp_b, -1, tmp_h // 2, tmp_w // 2)
                _,  pred_audio ,_ = self.netG(x=tmp, rev=True)
                pred_audio = istft_from_128x128_batch(pred_audio)

                
                self.extracted_audios.append(pred_audio.squeeze(0))
                self.changed_audios.append(attack_audio.squeeze(0))

            attack_audios = torch.cat(self.changed_audios, dim=0)
            pred_audios = torch.cat(self.extracted_audios, dim=0)

            self.orig_audio = self.orig_audio[:, :pred_audios.shape[0]]

            choice = 0

            if choice == 0:
                attack_audios = add_uniform_noise(attack_audios)
            elif choice == 1:
                attack_audios = add_echo(attack_audios, 16000)
            elif choice == 2:
                attack_audios = reduce_amplitude(attack_audios)

            self.extracted_audios = pred_audios
            self.changed_audios = attack_audios
            
            snr = calculate_snr(self.orig_audio, pred_audios.squeeze(0))
            pesq = calculate_pesq(self.orig_audio.squeeze(0), pred_audios.squeeze(0))   

            chunk_audio, T = self.divide_to_chunk(pred_audios.unsqueeze(0), 160)
            chunk_audio = self.moco.module.extract_features_q(chunk_audio)

            chunk_attack_audio, T = self.divide_to_chunk(attack_audios.unsqueeze(0), 160) #B, 32, 256
            chunk_attack_audio = self.moco.module.extract_features_q(chunk_attack_audio) #BT, 32

            

            similarity = F.cosine_similarity(chunk_audio, chunk_attack_audio, dim=1).view(-1, 1).squeeze()
            iou_similarity = similarity.view(-1, 1).repeat_interleave(160, dim=1).view(-1)
                
            self.mask = self.mask.view(-1)
    
            mask_t = self.mask.shape[0]
            mask_t = min(mask_t, iou_similarity.shape[0])

            self.mask = self.mask[:mask_t]
            iou_similarity = iou_similarity[:mask_t]

                
            thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            total_iou, self.pred = compute_IoU(iou_similarity, self.mask, thresholds)            
            auc = compute_AUC(iou_similarity, self.mask)
            ap = compute_AP(iou_similarity, self.mask)
            
                
            return auc, ap, total_iou, snr, pesq, self.extracted_audios, self.changed_audios

    def watermarking_test(self):
        self.netG.eval()
        self.moco.eval()

        torch.cuda.empty_cache()
        with torch.no_grad():
            self.containers = []
            self.audios = []

            vis_b, vis_t, vis_c, vis_h, vis_w = self.real_H.shape
            aud_b, aud_t, aud_sample = self.ref_L.shape

            total_psnr = 0.
            total_ssim = 0.

            for t in tqdm(range(vis_t), total=vis_t):
                hide_audio = self.ref_L[:, t] # B, 16000
                cover_video = self.real_H[:, t] # B, 3, 512, 512
                mask = self.mask[:, t]

                tmp = dwt(cover_video)

                stft_audio = stft_to_128x128_batch(hide_audio) #B, 128, 128, 2

                hide_data= hide_audio.view(aud_b, aud_sample, 1, 1).expand(aud_b, aud_sample, vis_h, vis_w)
                ones_tensor = torch.ones((aud_b, 1, vis_h, vis_w)).to(self.device)

                hide_data = torch.cat([ones_tensor, hide_data], dim = 1) 

                container, _ = self.netG(x=cover_audio, x_h=hide_data, rev=False)
                container = container.clamp(0, 1)

                psnr = calculate_psnr(container.squeeze(0).detach().cpu().numpy(), cover_video.squeeze(0).detach().cpu().numpy())
                ssim = calculate_ssim(container.squeeze(0).detach().cpu().numpy(), cover_video.squeeze(0).detach().cpu().numpy())

                total_psnr += psnr
                total_ssim += ssim

                self.containers.append(container)
                self.audios.append(hide_audio.squeeze(0))

            self.containers = torch.cat(self.containers, dim=0)
            self.audios = torch.cat(self.audios, dim=0)

            return vis_t, total_psnr, total_ssim, self.containers, self.audios, self.real_H.squeeze(0)

    def adjust_learning_rate(self, optimizer, epoch, total_epoch, train_opt):
        lr = train_opt['lr_E']
        if train_opt['cos']:
            lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / total_epoch))
        else:
            for milestone in train_opt['schedule']:
                lr *= 0.1 if epoch >= milestone else 1.0
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals_encoder(self):
        out_dict = OrderedDict()
        
        LR_ref = self.ref_L.detach()[0].float().cpu()
        out_dict['LR_ref'] = [video.squeeze(0) for video in LR_ref] #들어간 오디오
        
        out_dict['GT'] = self.real_H.detach()[0].float().cpu() #원래 이미지
        out_dict['LR'] = self.forw_L.detach().float().cpu() #삽입된 이미지
        
        SR = self.attack.detach()[0].float().cpu()
        out_dict['SR'] = [video.squeeze(0) for video in SR] #공격된 오디오
        
        
        return out_dict
    

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        load_path_M = self.opt['path']['pretrain_model_M']
        if load_path_G is not None:
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
            self.load_network(load_path_M, self.moco, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.moco, 'M', iter_label)

    def divide_to_chunk(self, audio, chunk_size):
        BT, sample = audio.shape
        T = sample // chunk_size

        audio = audio.view(BT, T, chunk_size).view(BT * T, 1, chunk_size)

        return audio, T
    
    def chunk_to_audio(self, audio, T):

        BT_T, channel = audio.shape

        audio = audio.view(BT_T//T, T, channel)
    
        return audio
    





