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
import time
import cv2

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


def main(json_path='options/test.json'):

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
        if phase == 'validation':
            val_set = define_Dataset(dataset_opt)
            if opt['rank'] == 0:
                val_dataset = DataLoader(val_set, batch_size=1,
                                     shuffle=dataset_opt['dataloader_shuffle'], num_workers=1,
                                     drop_last=False, pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_dataset = DataLoader(test_set, batch_size=1,
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
    model.load()  # load the model


    # if opt['rank'] == 0:
    #     logger.info(model.info_network())
    #     logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    
    avg_psnr_with_guide = 0
    avg_ssim_with_guide = 0
    avg_psnr_without_guide = 0
    avg_ssim_without_guide = 0
    avg_psnr_bicubic = 0
    avg_ssim_bicubic = 0

    it = 0
    if not opt["without_GT"]:
        for _,val_set in enumerate(val_dataset):
            image_name_ext = os.path.basename(val_set['Guide_path'][0])[:-4]
            print('Processing ', image_name_ext)
            if opt["without_guide"]:
                testdata_without_guide = val_set.copy()
                testdata_without_guide['Guide'] = torch.zeros(testdata_without_guide['Guide'].shape)
                save_guide_path = os.path.join(opt['path']['images'], '{:s}_withoutguide.bmp'.format(image_name_ext))
                model.feed_data(testdata_without_guide, phase='test')
                model.test()        
                visualwithout_guide = model.current_visuals(need_H=False)
                E_img_without_guide = util.tensor2uint(visualwithout_guide['Output'])
                    
                util.imsave(E_img_without_guide, save_guide_path)
            
            save_img_path = os.path.join(opt['path']['images'], '{:s}.bmp'.format(image_name_ext))
            model.feed_data(val_set, phase='test', need_GT=True)
            model.test()
            visuals = model.current_visuals(need_H=True)
            E_img = util.tensor2uint(visuals['Output'])
            #E_img = cv2.applyColorMap(E_img, cv2.COLORMAP_JET)
            util.imsave(E_img, save_img_path)

            Lr_image = util.tensor2uint(visuals['Lr'])
            save_Lr_path = os.path.join(opt['path']['images'], '{:s}_lr.bmp'.format(image_name_ext))
            util.imsave(Lr_image, save_Lr_path)

            H_img = util.tensor2uint(visuals['GT'])
            psnr_without_guide = util.calculate_psnr(E_img_without_guide, H_img)
            ssim_psnr_without_guide = util.calculate_ssim(E_img_without_guide, H_img)
            avg_psnr_without_guide += psnr_without_guide
            avg_ssim_without_guide += ssim_psnr_without_guide

            psnr_with_guide = util.calculate_psnr(E_img, H_img)
            ssim_psnr_with_guide = util.calculate_ssim(E_img, H_img)
            avg_psnr_with_guide += psnr_with_guide
            avg_ssim_with_guide += ssim_psnr_with_guide

            psnr_bicubic = util.calculate_psnr(Lr_image, H_img)
            ssim_bicubic = util.calculate_ssim(Lr_image, H_img)
            avg_psnr_bicubic += psnr_bicubic
            avg_ssim_bicubic += ssim_bicubic
            it += 1
        avg_psnr_with_guide /= it
        avg_ssim_with_guide /= it
        avg_psnr_without_guide /= it
        avg_ssim_without_guide /= it
        avg_psnr_bicubic /= it
        avg_ssim_bicubic /= it

        print('Average PSNR without guide: ', avg_psnr_without_guide)
        print('Average SSIM without guide: ', avg_ssim_without_guide)
        print('Average PSNR with guide: ', avg_psnr_with_guide)
        print('Average SSIM with guide: ', avg_ssim_with_guide)
        print("percentage degradation in PSNR: ", (avg_psnr_without_guide-avg_psnr_with_guide )/avg_psnr_without_guide * 100)
        print("percentage degradation in SSIM: ", (avg_ssim_without_guide-avg_ssim_with_guide )/avg_ssim_without_guide * 100)
        print('Average PSNR bicubic: ', avg_psnr_bicubic)
        print('Average SSIM bicubic: ', avg_ssim_bicubic)
    else:
        avg_time = 0
        it = 0 
        for _,test_set in enumerate(test_dataset):
            print('Processing ', test_set['Guide_path'])
            image_name_ext = os.path.basename(test_set['Guide_path'][0])[:-4]

            save_img_path = os.path.join(opt['path']['images'], '{:s}.png'.format(image_name_ext))
            start_time = time.time()
            #print(test_set['Guide_path'].shape)
            model.feed_data(test_set, phase='test', need_GT=False)
            model.test()
            end_time = time.time() - start_time
            avg_time += end_time
            it += 1
            visuals = model.current_visuals(need_H=False)
            E_img = util.tensor2uint(visuals['Output'])
            E_img = cv2.resize(E_img,dsize=(960,1280), interpolation=cv2.INTER_CUBIC)
            util.imsave(E_img, save_img_path)
        print('Average time: ', avg_time/it)

  
  



if __name__ == '__main__':
    main()
