import glob
import cv2
import random
import numpy as np
import pickle
import os
from torch.utils import data

def crop(config, x, t, M):
    crop_h = config.height
    crop_w = config.width
    c, h, w = x.shape
    l = random.randint(0, w - 1 - crop_w)
    u = random.randint(0, h - 1 - crop_h)
    
    return x[:, u: u+crop_h, l: l+crop_w], t[:, u: u+crop_h, l: l+crop_w], M[u: u+crop_h, l: l+crop_w]

def critical_crop(config, x, t, M):
    half_crop_h = int(config.height/2)
    half_crop_w = int(config.width/2)
    c, h, w = x.shape

    shadow = np.nonzero(M)

    idx = random.randint(0, len(shadow[0]) - 1)
    center_h = int(shadow[0][idx])
    center_w = int(shadow[1][idx])
    # print(center_h, center_w)
    top = center_h - half_crop_h
    b = center_h + half_crop_h
    l = center_w - half_crop_w
    r = center_w + half_crop_w
    # print(h, w)
    # print(top, b, l, r)
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
    # print(l, r, t, b)
    # print(top, b, l, r)
    return x[:, top:b, l:r], t[:, top:b, l:r], M[top:b, l:r]


class TrainDataset(data.Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config
       
        train_list_file = os.path.join(config.datasets_dir, config.train_list)
        print(train_list_file)
        if not os.path.exists(train_list_file) or os.path.getsize(train_list_file) == 0:
            files = os.listdir(os.path.join(config.datasets_dir, 'train_C'))
            n_train = len(files)
            train_list = files[1280:n_train]
            np.savetxt(os.path.join(config.datasets_dir, config.train_list), np.array(train_list), fmt='%s')

        self.imlist = np.loadtxt(train_list_file, str)  
        print(self.imlist)
    def __getitem__(self, index):
        
        t = cv2.imread(os.path.join(self.config.datasets_dir, 'train_C', str(self.imlist[index])), 1).astype(np.float32)
        x = cv2.imread(os.path.join(self.config.datasets_dir, 'train_A', str(self.imlist[index])), 1).astype(np.float32)

        M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)
        # print(x.shape)
        x, t, M = critical_crop(self.config, x, t, M)
        # print(x.shape)
        # print(x.shape, t.shape, M.shape)
        return x, t, M

    def __len__(self):
        return len(self.imlist)

    
    
class ValDataset(data.Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config

        val_list_file = os.path.join(config.valset_dir, config.validation_list)
        # 如果数据集尚未分割，则进行训练集和测试集的分割
        # if not os.path.exists(val_list_file) or os.path.getsize(val_list_file) == 0:
        #     files = os.listdir(os.path.join(config.valset_dir, 'test_C'))
        #     n_val = len(files)
        #     val_list = files[:n_val]
        #     np.savetxt(os.path.join(config.valset_dir, config.validation_list), np.array(val_list), fmt='%s')

        self.imlist = np.loadtxt(val_list_file, str)
        print(self.imlist)

    def __getitem__(self, index):
        
        t = cv2.imread(os.path.join(self.config.valset_dir, 'test_C', str(self.imlist[index])), 1).astype(np.float32)
        x = cv2.imread(os.path.join(self.config.valset_dir, 'test_A', str(self.imlist[index])), 1).astype(np.float32)
        # t = cv2.resize(t, (self.config.height, self.config.width), interpolation=cv2.INTER_CUBIC)
        # x = cv2.resize(x, (self.config.height, self.config.width), interpolation=cv2.INTER_CUBIC)
        
        M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)
        # print(x.shape, t.shape, X.shape)
        # x, t, M = crop(self.config, x, t, M)
        return x, t, M
        

    def __len__(self):
        return len(self.imlist)    
    
    
    


class TestDataset(data.Dataset):
    def __init__(self, test_dir, in_ch, out_ch):
        super().__init__()
        self.test_dir = test_dir
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.test_files = os.listdir(os.path.join(test_dir))

    def __getitem__(self, index):
        filename = os.path.basename(self.test_files[index])
        #print("filename=", filename)
        x = cv2.imread(os.path.join(self.test_dir, filename), 1)
        #print("A:",x)

        x = x.astype(np.float32)
        #print("a:",x)

        x = x / 255
       #print(x)

        x = x.transpose(2, 0, 1)
        #print(x)

        return x, filename

    def __len__(self):

        return len(self.test_files)
