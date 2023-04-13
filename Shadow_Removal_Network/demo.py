import numpy as np
import argparse
from cv2 import cv2
import matplotlib.pyplot as plt
import time

import torch
from torch.autograd import Variable
from data_manager import TestDataset
from utils import gpu_manage, heatmap, save_image
# from models.gen.SPANet import Generator
from SpA_Former import Generator
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict(args):
    test_dir = "/mnt/188/b/NTIRE23/data/shadow/ISTD/ISTD_Dataset/test/test_A"
    gpu_manage(args)
    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=args.gpu_ids)
    param = torch.load("/mnt/188/b/NTIRE23/data/shadow/SpA-Former-shadow-removal/weights/gen_model_epoch_160.pth")
    # param_2 = torch.load("/mnt/188/b/NTIRE23/data/shadow/SpA-Former-shadow-removal/results_ISTD/000158/models/gen_model_epoch_759.pth")
    # print(param_1.keys())
    # print(param_2.keys())
    # s1 = set(param_1.keys())
    # s2 = set(param_2.keys())
    # print(s2 - s1)
    # sys.exit()
    gen.load_state_dict(param)
    gen = gen.cuda(0)
    gen.eval()
    validation_dataset = TestDataset(test_dir, 3, 3)
    validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False)
    

    print ('<=== Model loaded')
    with torch.no_grad():
        for _, batch in enumerate(tqdm(validation_data_loader)):
            x, filename = Variable(batch[0]), batch[1]
            x = x.cuda(0)
            att, out = gen(x)
            out_ = out.cpu().numpy()[0]
            out_rgb = np.clip(out_[:3], 0, 1)
            out_rgb *= 255
            out_rgb = out_rgb.transpose(1, 2, 0)
            cv2.imwrite(f'/mnt/188/b/NTIRE23/data/shadow/ISTD/ISTD_Dataset/test/pseudo_free/{filename[0]}', out_rgb)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_filepath', type=str, required=True)
    # parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)
    args = parser.parse_args()

    predict(args)