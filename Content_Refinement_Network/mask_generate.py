
import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from ptflops import get_model_complexity_info

import scipy.io as sio
from utils.loader import get_train_data , get_train_data_ISTD, get_test_data_ISTD
import utils
import cv2
from model import UNet

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from sklearn.metrics import mean_squared_error as mse_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='../ISTD_Dataset/test/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='/mnt/188/b/NTIRE23/data/shadow/ShadowFormer/log/ShadowFormer_istd/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='ShadowFormer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--cal_metrics', action='store_true', help='Measure denoised images with GT')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=10, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
parser.add_argument('--train_ps', type=int, default=320, help='patch size of training sample')
parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
test_dataset = get_train_data_ISTD()
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

img_multiple_of = 8 * args.win_size
maes = 0
with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):

        mask = data_test[2].numpy()
        # mask = mask.unsqueeze(1)
        filenames = data_test[3]
        # mask = np.repeat(mask, 3, axis=0)
        
        cv2.imwrite(f'/mnt/188/b/NTIRE23/data/shadow/ISTD/ISTD_Dataset/train/pseudo_mask/{filenames[0]}', mask[0]* 255.0)
# print(maes / len(test_loader))

