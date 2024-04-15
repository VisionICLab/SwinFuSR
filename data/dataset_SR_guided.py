import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util

import logging
from utils import utils_logger

import cv2


class Dataset(data.Dataset): 
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """
    def __init__(self, opt):
        super(Dataset, self).__init__()
        print('Dataset: for guided super-resolution.')
        self.opt = opt
        self.n_channels_lr = opt['n_channels_lr'] if opt['n_channels_lr'] else 1
        self.n_channels_guide = opt['n_channels_guide'] if opt['n_channels_guide'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4

        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.L_size = self.patch_size // self.sf
        self.patch = opt["patch"]
        self.proba_without_rgb = opt['proba_without_rgb']

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_lr = util.get_image_paths(opt['dataroot_lr'])
        self.path_guide = util.get_image_paths(opt['dataroot_guide'])
        self.path_gt = util.get_image_paths(opt['dataroot_gt'])


    def __getitem__(self, index):

        Hr_path = self.path_gt[index]
        img_hr = util.imread_uint(Hr_path, self.n_channels_lr)
        Lr_path = None

        if self.paths_lr:
            # --------------------------------
            # directly load L image
            # --------------------------------
            Lr_path = self.paths_lr[index]
            img_Lr = util.imread_uint(Lr_path, self.n_channels_lr)
            img_Lr = np.clip(util.imresize_np(img_Lr, self.sf, True),0,255)
        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_hr.shape[:2]
            img_Lr = util.imresize_np(img_hr, 1 / self.sf, True)
            img_Lr = util.imresize_np(img_Lr, self.sf, True)

        if self.path_guide:
            Guide_path = self.path_guide[index]
            img_guide = util.imread_uint(Guide_path, self.n_channels_guide)

        else:
            # case where the guide image is missing
            Guide_path = Hr_path
            img_guide = img_Lr


        img_guide = util.imread_uint(Guide_path, self.n_channels_guide)
        if len(img_guide.shape)==3:
            img_guide = cv2.cvtColor(img_guide, cv2.COLOR_RGB2YCrCb)
            img_guide = np.expand_dims(img_guide[:,:,0], axis=2)  # HxWx1

        if self.opt['phase'] == 'test': 
            #print(img_guide.shape,img_hr.shape)
            target = (640,448)
            img_Lr,img_guide = util.uint2tensor3(cv2.resize(img_hr,dsize=target, interpolation=cv2.INTER_CUBIC)),util.uint2tensor3(cv2.resize(img_guide,dsize=target, interpolation=cv2.INTER_CUBIC))
            #print(img_Lr.shape,img_guide.shape)
            return {'Guide': img_guide, 'Lr': img_Lr, 'Guide_path': Guide_path, 'Lr_path': Hr_path}

        
        
        """
        if index%10 == 0:
            util.imsave(img_Lr,f"/home/travail/Code/SwinFuSR/Model/SR_competition/Guided SR/images/lr{index}.png")
            util.imsave(img_guide,f"/home/travail/Code/SwinFuSR/Model/SR_competition/Guided SR/images/guide{index}.png")
            util.imsave(img_hr,f"/home/travail/Code/SwinFuSR/Model/SR_competition/Guided SR/images/hr{index}.png")
        """
        if self.opt['phase'] == 'train'and self.patch: 
            """
            # --------------------------------
            # get under/over/norm patch pairs
            # --------------------------------
            """
            H, W, _ = img_Lr.shape
            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_lr = img_Lr[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_guide = img_guide[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_hr = img_hr[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            
          
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0,7)
            # print('img_A shape:', img_A.shape)
            
            img_Lr, img_guide,img_hr = util.augment_img(patch_lr, mode=mode), util.augment_img(patch_guide, mode=mode),util.augment_img(patch_hr, mode=mode)
            if self.proba_without_rgb>0:
                if random.random() < self.proba_without_rgb:
                    img_guide = torch.zeros(img_guide.shape)
        """
        if index%10 == 0:
            util.imsave(patch_lr,f"/home/travail/Code/SwinFuSR/Model/SR_competition/Guided SR/images/patch_lr{index}.png")
            util.imsave(patch_guide,f"/home/travail/Code/SwinFuSR/Model/SR_competition/Guided SR/images/patch_guide{index}.png")
            util.imsave(patch_hr,f"/home/travail/Code/SwinFuSR/Model/SR_competition/Guided SR/images/patch_hr{index}.png")
        if index%10 == 0:
            util.imsave(img_Lr,f"/home/travail/Code/SwinFuSR/Model/SR_competition/Guided SR/images/patch_lr_augmented{index}.png")
            util.imsave(img_guide,f"/home/travail/Code/SwinFuSR/Model/SR_competition/Guided SR/images/patch_guide_augmented{index}.png")
            util.imsave(img_hr,f"/home/travail/Code/SwinFuSR/Model/SR_competition/Guided SR/images/patch_hr_augmented{index}.png")
        """

        img_Lr, img_guide,img_hr = util.uint2tensor3(img_Lr),util.uint2tensor3(img_guide),util.uint2tensor3(img_hr)
      
        if Lr_path is None:
            Lr_path = Hr_path
        #print("Lr size",img_Lr.size(),"Guide size",img_guide.size(),"Hr size",img_hr.size())
        return {'Lr': img_Lr, 'Guide': img_guide, 'Hr': img_hr, 'Lr_path': Lr_path, 'Guide_path': Guide_path, 'Hr_path': Hr_path}    

    def __len__(self):
        return len(self.path_gt)
