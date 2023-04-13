import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, load_val_img, load_mask, load_val_mask, Augment_RGB_torch
import torch.nn.functional as F
import random
import cv2
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 
def critical_crop(config, x, t, M):
    half_crop_h = int(config.train_ps/2)
    half_crop_w = int(config.train_ps/2)
    c, h, w = x.shape
    # print(h, w)
    shadow = np.nonzero(M)
    idx = random.randint(0, len(shadow[0]) - 1)
    center_h = int(shadow[0][idx])
    center_w = int(shadow[1][idx])

    top = center_h - half_crop_h
    b = center_h + half_crop_h
    l = center_w - half_crop_w
    r = center_w + half_crop_w
    # print(l, r, top, b)
    if l < 0:
        r -= l
        l = 0
    if top < 0:
        b -= top
        top = 0
    if b >= h:
        top -= (b - h)
        b = h
    if r >= w:
        l -= (r - w)
        r = w
    # print(l, r, top, b)
    return x[:, top:b, l:r], t[:, top:b, l:r], M[top:b, l:r]
##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, config):
        super(DataLoaderTrain, self).__init__()

        self.config = config
        train_list_file = os.path.join(config.datasets_dir, config.train_list)
        self.imlist = np.loadtxt(train_list_file, str) 

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, index):
        t = cv2.imread(os.path.join(self.config.datasets_dir, 'train_C', str(self.imlist[index])), 1).astype(np.float32)
        x = cv2.imread(os.path.join(self.config.datasets_dir, 'train_A', str(self.imlist[index])), 1).astype(np.float32)
        # pseudo_free = cv2.imread(os.path.join(self.config.datasets_dir, 'train_C', str(self.imlist[index])), 1).astype(np.float32)
        pseudo_free = cv2.imread(os.path.join(self.config.datasets_dir, 'pseudo_free', str(self.imlist[index])), 1).astype(np.float32)

        M = np.clip((pseudo_free-x).sum(axis=2), 0, 1).astype(np.float32)

        # M = np.clip((pseudo_free-x).sum(axis=2), 0, 1).astype(np.float32)
        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)
        
        # x, t, M = critical_crop(self.config, x, t, M)
        return t, x, M, '', ''
        # return clean, noisy, mask, clean_filename, noisy_filename

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, config):
        super(DataLoaderVal, self).__init__()
        self.config = config
        val_list_file = os.path.join(config.valset_dir, config.validation_list)
        self.imlist = np.loadtxt(val_list_file, str)
        
    def __len__(self):
        return len(self.imlist)    

    def __getitem__(self, index):
        t = cv2.imread(os.path.join(self.config.valset_dir, 'test_C', str(self.imlist[index])), 1).astype(np.float32)
        x = cv2.imread(os.path.join(self.config.valset_dir, 'test_A', str(self.imlist[index])), 1).astype(np.float32)
        # pseudo_free = cv2.imread(os.path.join(self.config.valset_dir, 'test_C', str(self.imlist[index])), 1).astype(np.float32)
        pseudo_free = cv2.imread(os.path.join(self.config.valset_dir, 'pseudo_free', str(self.imlist[index])), 1).astype(np.float32)

        M = np.clip((pseudo_free-x).sum(axis=2), 0, 1).astype(np.float32)
        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)

        return t, x, M, '', ''

class DataLoaderTest(Dataset):
    def __init__(self, test_dir, pseudo_gt_dir):
        super().__init__()
        print(test_dir, pseudo_gt_dir)
        self.test_dir = test_dir
        self.pseudo_gt_dir = pseudo_gt_dir
        self.test_files = os.listdir(os.path.join(test_dir))

    def __getitem__(self, index):
        filename = os.path.basename(self.test_files[index])
        # print(os.path.join(self.pseudo_gt_dir, filename))
        t = cv2.imread(os.path.join(self.pseudo_gt_dir, filename), 1).astype(np.float32)
        x = cv2.imread(os.path.join(self.test_dir, filename), 1).astype(np.float32)
        M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)

        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)

        return x, t, M, filename

    def __len__(self):

        return len(self.test_files)

class DataLoaderTestMask(Dataset):
    def __init__(self, test_dir, gt_dir, pseudo_gt_dir):
        super().__init__()
        self.test_dir = test_dir
        self.gt_dir = gt_dir
        self.pseudo_gt_dir = pseudo_gt_dir
        self.test_files = os.listdir(os.path.join(test_dir))

    def __getitem__(self, index):
        filename = os.path.basename(self.test_files[index])
        # print(os.path.join(self.pseudo_gt_dir, filename))
        t_gt = cv2.imread(os.path.join(self.gt_dir, filename), 1).astype(np.float32)
        t_pseudo = cv2.imread(os.path.join(self.pseudo_gt_dir, filename), 1).astype(np.float32)
        x = cv2.imread(os.path.join(self.test_dir, filename), 1).astype(np.float32)
        
        M_t = np.clip((t_gt-x).sum(axis=2), 0, 1).astype(np.float32)
        M_p = np.clip((t_pseudo-x).sum(axis=2), 0, 1).astype(np.float32)
        x = x / 255
        # t = t / 255
        x = x.transpose(2, 0, 1)
        # t = t.transpose(2, 0, 1)

        return x, "None", M_t, M_p, filename

    def __len__(self):

        return len(self.test_files)
