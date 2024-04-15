import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torchsummary import summary

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
import wandb

from data.select_dataset import define_Dataset
from models.select_model import define_Model
import warnings
warnings.filterwarnings("ignore")


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_baseline.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        for key, path in opt['path'].items():
            print(path)
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')

    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))
    if opt["wandb"]:
        wandb.init(project="SwinFuSR", 
               name="SwinFuSR", 
               config=opt,
               dir=opt['path']['log'])
    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / opt["train"]['batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=opt["train"]['batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=opt["train"]['batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)
        elif phase == 'validation':
            val_set = define_Dataset(dataset_opt)
            if opt['rank'] == 0:
                logger.info('Number of validaton images: {:,d}'.format(len(val_set)))
            val_loader = DataLoader(val_set, batch_size=1,
                                     shuffle=dataset_opt['dataloader_shuffle'], num_workers=1,
                                     drop_last=False, pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()

    summary(model.netG, [(1, 64, 64),(1, 64, 64)], opt['train']['batch_size']//len(opt['gpu_ids']))


    # if opt['rank'] == 0:
    #     logger.info(model.info_network())
    #     logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    need_GT = True
    max_psnr = 0
    for epoch in range(opt["train"]["epochs"]):  # keep running
        #start_time_epoch = time.time()

       
        #start_loop =  time.time()
        for i, train_data in enumerate(train_loader):
            #end_loop =  time.time()
            #elapsed_time = end_loop - start_loop
            #print(f"Time loop: {elapsed_time:.2f} seconds")

            #start_time_it = time.time()
            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data, need_GT=need_GT)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step,phase="train")
            
             # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()
                message = '<epoch:{}, iter:{:}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)
                logs = {'epoch':epoch, 'iter':current_step, 'lr':model.current_learning_rate()}|logs
                if opt["wandb"]:
                    wandb.log(logs)


            # -------------------------------
            # 4) validation information
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:
                iter_num = 0
                avg_psnr = 0
                avg_ssim = 0
                for it,validation_data in enumerate(val_loader):
                    if it > opt['train']['limit_validation'] and epoch<20000:
                            break
                    model.feed_data(validation_data, phase='validation', need_GT=True)
                    model.test()
                    visuals = model.current_visuals(need_H=True)
                    E_img = util.tensor2uint(visuals['Output'])
                    H_img = util.tensor2uint(visuals['GT'])
                    current_psnr = util.calculate_psnr(E_img, H_img)
                    current_ssim = util.calculate_ssim(E_img, H_img)
                    avg_ssim += current_ssim
                    avg_psnr += current_psnr
                    iter_num += 1
                avg_psnr = avg_psnr / iter_num
                avg_ssim = avg_ssim / iter_num
                metrics = {'avg_psnr':avg_psnr, 'avg_ssim':avg_ssim}|{k: v for k, v in logs.items()}|{'epoch':epoch, 'iter':current_step, 'lr':model.current_learning_rate()}
                logger.info('<epoch:{}, iter:{}, Average PSNR : {:<.2f}dB, Average SSIM : {:<.4f}\n'.format(epoch, current_step, avg_psnr, avg_ssim))
                if avg_psnr > max_psnr:
                    max_psnr = avg_psnr
                    if opt["wandb"]:
                        wandb.run.summary["max_psnr"] = max_psnr
                        wandb.run.summary["ssim"] = avg_ssim
                    save_dir = opt['path']['root'] 
                    save_filename = '{}_{}.pth'.format(current_step, epoch)
                    save_path = os.path.join(save_dir, save_filename)
                    logger.info('Saving the model. Save path is:{}'.format(save_path))
                    model.save(current_step)
                    table = wandb.Table(columns=["prediction"])
                    for it,test_data in enumerate(test_loader):
                        if it > opt['train']['limit_test']:
                            break
                        image_name_ext = os.path.basename(test_data['Guide_path'][0])
                        #print("image_name_ext",image_name_ext)

                        model.feed_data(test_data, phase='test')
                        model.test()
                        visuals = model.current_visuals(need_H=False)
                        E_img = util.tensor2uint(visuals['Output'])
                        #E_img = cv2.applyColorMap(E_img, cv2.COLORMAP_JET)

                        # -----------------------
                        # save estimated image E
                        # -----------------------
                        save_img_path = os.path.join(opt['path']['images'], '{:s}'.format(image_name_ext))
                        util.imsave(E_img, save_img_path)
                        table.add_data(wandb.Image(E_img))
                    metrics = metrics|{"Table":table}
                if opt["wandb"]:
                    wandb.log(metrics)
            #end_time_it = time.time()
            #elapsed_time = end_time_it - start_time_it
            #print(f"Time iteratiion: {elapsed_time:.2f} seconds")
        #end_time_epoch = time.time()
        #elapsed_time = end_time_epoch - start_time_epoch
        #print(f"Time epoch: {elapsed_time:.2f} seconds")



if __name__ == '__main__':
    main()
