import os
import math
import argparse
import random
import logging
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from torch.backends import cudnn

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from models.metric.metric import calculate_snr, calculate_ssim, calculate_psnr


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def main():
    import wandb
    
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.', default="./options/test.yml")  # config 文件
    parser.add_argument('--mask_type', choices = ['random', 'facial'], default='facial', help='Masking Strategy')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=10)

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0  

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

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    elif opt['path'].get('pretrain_model_G', None):
        device_id = torch.cuda.current_device()
        resume_state = None
    else:
        resume_state = None

    # mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 1  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            opt['total_epochs'] = total_epochs
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # create model
    model = create_model(opt)
    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            # training
            model.feed_data(train_data)
            model.optimize_parameters(epoch=epoch, train_type='all')

            # update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if (current_step) % opt['train']['val_freq'] == 0 and rank <= 0:
                #if rank <= 0:
                avg_snr = 0.0
                idx = 0
                total_iou = [0.0 for _ in range(10)]
                total_idx = 0
                total_frame = 0.0
                total_auc = 0.0
                total_psnr = 0.0
                total_ssim = 0.0
                for video_id, val_data in enumerate(val_loader):
                    print(video_id)
                    img_dir = os.path.join(opt['path']['val_images'])
                    util.mkdir(img_dir)
                    
                    model.feed_data(val_data)
                    iou_results, auc = model.test(wandb, test_type='encoder')

                    total_frame += 1
                    total_auc += auc

                    for i in range(10):
                        total_iou[i] += iou_results[i]

                    visuals = model.get_current_visuals_encoder()

                    gt_aud = util.tensor2audio(visuals['LR_ref'][0].squeeze())
                    sr_aud = util.tensor2audio(visuals['SR'][0].squeeze())


                    t_step = len(visuals['LR'])
                    total_idx += t_step
                    idx += t_step

                    save_aud_path = os.path.join(img_dir,'{:d}_{:s}.wav'.format(video_id, 'LR_ref'))
                    #util.save_audio(gt_aud, save_aud_path)
                        
                    save_aud_path = os.path.join(img_dir,'{:d}_{:s}.wav'.format(video_id, 'SR'))
                    #util.save_audio(sr_aud, save_aud_path)

                    for i in range(t_step):

                        gt_img = util.tensor2img(visuals['GT'][i])  
                        lr_img = util.tensor2img(visuals['LR'][i])

                        print(gt_img.shape)
                        print(lr_img.shape)

                        gt_img_for_metric = util.tensor2metric(visuals['GT'][i])
                        lr_img_for_metric = util.tensor2metric(visuals['LR'][i])

                        ssim = calculate_ssim(gt_img, lr_img)
                        psnr = calculate_psnr(gt_img, lr_img)

                        total_psnr += psnr
                        total_ssim += ssim
                        

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(video_id, i, 'GT'))
                        util.save_img(gt_img, save_img_path)

                        save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(video_id, i, 'LR'))
                        util.save_img(lr_img, save_img_path)





                    

                N = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                logger.info({"AUC": total_auc/total_frame, "PSNR": total_psnr/total_idx, "SSIM" : total_ssim/total_idx, "IoU0.0": total_iou[0]/total_frame, "IoU0.1": total_iou[1]/total_frame, "IoU0.2": total_iou[2]/total_frame, "IoU0.3": total_iou[3]/total_frame, "IoU0.4": total_iou[4]/total_frame, "IoU0.5": total_iou[5]/total_frame, "IoU0.6": total_iou[6]/total_frame, "IoU0.7": total_iou[7]/total_frame, "IoU0.8": total_iou[8]/total_frame, "IoU0.9": total_iou[9]/total_frame})

                wandb.log({"AUC": total_auc/total_frame,  "PSNR": total_psnr/total_idx, "SSIM" : total_ssim/total_idx, "IoU0.0": total_iou[0]/total_frame, "IoU0.1": total_iou[1]/total_frame, "IoU0.2": total_iou[2]/total_frame, "IoU0.3": total_iou[3]/total_frame, "IoU0.4": total_iou[4]/total_frame, "IoU0.5": total_iou[5]/total_frame, "IoU0.6": total_iou[6]/total_frame, "IoU0.7": total_iou[7]/total_frame, "IoU0.8": total_iou[8]/total_frame, "IoU0.9": total_iou[9]/total_frame})


            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
