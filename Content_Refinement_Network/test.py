
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
from model_local1 import ShadowFormer_base
import scipy.io as sio
from utils.loader import get_test_data, get_train_data, get_val_data, get_test_data_ISTD
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
parser.add_argument('--weights', default='/mnt/188/b/NTIRE23/data/shadow/ShadowFormer/log_real_train_pseudo_val/ShadowFormer_istd/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='ShadowFormer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--cal_metrics', action='store_true', help='Measure denoised images with GT')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=10, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--split', type=str,default='val', help='ffn/leff token mlp')
parser.add_argument('--ver', type=str,default='0', help='version of mask')

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

utils.mkdir(args.result_dir)
# if args.split == 'val':
#     test_dataset = get_val_data(args.ver)
# elif args.split == 'test':
#     test_dataset = get_test_data(args.ver)
# elif args.split == 'train':
#     test_dataset = get_train_data(args.ver)
test_dataset = get_test_data_ISTD()
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

# model_restoration = utils.get_arch(args)
# model_restoration = torch.nn.DataParallel(model_restoration)
# utils.load_checkpoint(model_restoration, args.weights)
model_restoration = ShadowFormer_base(img_size=args.train_ps,embed_dim=args.embed_dim,win_size=args.win_size,token_projection=args.token_projection,token_mlp=args.token_mlp)
model_restoration = torch.nn.DataParallel(model_restoration)

model_restoration.load_state_dict(torch.load(args.weights)['state_dict'])
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

img_multiple_of = 8 * args.win_size

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    rmse_val_rgb = []
    psnr_val_s = []
    ssim_val_s = []
    psnr_val_ns = []
    ssim_val_ns = []
    rmse_val_s = []
    rmse_val_ns = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):

        input_ = data_test[0].cuda()
        mask = data_test[2].cuda()
        mask = mask.unsqueeze(1)
        filenames = data_test[3]
        
        restored = model_restoration(input_, mask)

        rgb_restored = torch.clamp(restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

        # if args.save_images:
        utils.save_img(rgb_restored*255.0, os.path.join(args.result_dir, filenames[0]))

if args.cal_metrics:
    psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
    ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
    psnr_val_s = sum(psnr_val_s)/len(test_dataset)
    ssim_val_s = sum(ssim_val_s)/len(test_dataset)
    psnr_val_ns = sum(psnr_val_ns)/len(test_dataset)
    ssim_val_ns = sum(ssim_val_ns)/len(test_dataset)
    rmse_val_rgb = sum(rmse_val_rgb) / len(test_dataset)
    rmse_val_s = sum(rmse_val_s) / len(test_dataset)
    rmse_val_ns = sum(rmse_val_ns) / len(test_dataset)
    print("PSNR: %f, SSIM: %f, RMSE: %f " %(psnr_val_rgb, ssim_val_rgb, rmse_val_rgb))
    print("SPSNR: %f, SSSIM: %f, SRMSE: %f " %(psnr_val_s, ssim_val_s, rmse_val_s))
    print("NSPSNR: %f, NSSSIM: %f, NSRMSE: %f " %(psnr_val_ns, ssim_val_ns, rmse_val_ns))

