import os
import random
import shutil
import yaml
from attrdict import AttrMap
import time

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from data_manager import TrainDataset,ValDataset
from SpA_Former import Generator
from models.dis.dis import Discriminator
import utils
from utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport
import sys
from tqdm import tqdm
import numpy as np
from utils import save_image
from thop import profile

seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
def train(config):
    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')
    
    train_dataset = TrainDataset(config)
    validation_dataset = ValDataset(config)
    print('train dataset:', len(train_dataset))
    print('validation dataset:', len(validation_dataset))
    # sys.exit(0)
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize, shuffle=True,  drop_last=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, batch_size=config.validation_batchsize, shuffle=False)
    
    
    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)
    flops, params = profile(gen, inputs=(torch.randn(1, 3, 360,480),))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    # sys.exit()
    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)
    flops, params = profile(dis, inputs=(torch.randn(1, 6, 360,480),))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    if config.dis_init is not None:
        param = torch.load(config.dis_init)
        dis.load_state_dict(param)
        print('load {} as pretrained model'.format(config.dis_init))

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.gen_lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    opt_dis = optim.Adam(dis.parameters(), lr=config.dis_lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    real_a = torch.FloatTensor(config.batchsize, config.in_ch, config.width, config.height)
    real_b = torch.FloatTensor(config.batchsize, config.out_ch, config.width, config.height)
    M = torch.FloatTensor(config.batchsize, config.width, config.height)

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()

    if config.cuda:
        gen = gen.cuda()
        dis = dis.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        criterionSoftplus = criterionSoftplus.cuda()
        real_a = real_a.cuda()
        real_b = real_b.cuda()
        M = M.cuda()

    real_a = Variable(real_a)
    real_b = Variable(real_b)

    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    print('===> begin')
    start_time=time.time()
    # main
    it = 0
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()

        gen.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a_cpu, real_b_cpu, M_cpu = batch
            real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
            real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)
            M.resize_(M_cpu.size()).copy_(M_cpu)
            # print(real_b.size(), real_a.size())
            
            att, fake_b = gen.forward(real_a)
            if iteration == 0:
                h = 1
                w = 3
                c = 3
                p = x.size(2)
                q = x.size(3)

                allim = np.zeros((h, w, c, p, q))
                x_ = x.cpu().numpy()[0]
                t_ = t.cpu().numpy()[0]
                out_ = fake_b.cpu().numpy()[0]
                in_rgb = x_[:3]
                t_rgb = t_[:3]
                out_rgb = np.clip(out_[:3], 0, 1)
                allim[0, 0, :] = in_rgb * 255
                allim[0, 1, :] = out_rgb * 255
                allim[0, 2, :] = t_rgb * 255
                
                allim = allim.transpose(0, 3, 1, 4, 2)
                allim = allim.reshape((h*p, w*q, c))

                save_image(config.out_dir, allim, i, epoch, filename='sample.png')
            ################
            ### Update D ###
            ################
            
            opt_dis.zero_grad()

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab.detach())
            batchsize, _, w, h = pred_fake.size()

            loss_d_fake = torch.sum(criterionSoftplus(pred_fake)) / batchsize / w / h

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = dis.forward(real_ab)
            loss_d_real = torch.sum(criterionSoftplus(-pred_real)) / batchsize / w / h

            # Combined loss
            loss_d = loss_d_fake + loss_d_real

            loss_d.backward()

            if epoch % config.minimax == 0:
                opt_dis.step()

            ################
            ### Update G ###
            ################
            if it % config.g_freq == 0:
                opt_gen.zero_grad()

                # First, G(A) should fake the discriminator
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = dis.forward(fake_ab)
                loss_g_gan = torch.sum(criterionSoftplus(-pred_fake)) / batchsize / w / h

                # Second, G(A) = B
                loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb
                loss_g_att = criterionMSE(att[:,0,:,:], M)
                loss_g = loss_g_gan + loss_g_l1 + loss_g_att
                # loss_g = loss_g_l1 + loss_g_att

                loss_g.backward()

                opt_gen.step()
            it += 1
            # log
            if iteration % 10 == 0:
                print("===> Epoch[{}]({}/{}): loss_d_fake: {:.4f} loss_d_real: {:.4f} loss_g_gan: {:.4f} loss_g_l1: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d_fake.item(), loss_d_real.item(), loss_g_gan.item(), loss_g_l1.item()))
                
                log = {}
                log['epoch'] = epoch
                log['iteration'] = len(training_data_loader) * (epoch-1) + iteration
                log['gen/loss'] = loss_g.item()
                log['dis/loss'] = loss_d.item()

                logreport(log)

        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        gen.eval()
        with torch.no_grad():
            log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
            validationreport(log_validation)
        print('validation finished')
        if epoch % config.snapshot_interval == 0:
            checkpoint(config, epoch, gen, dis)

        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    print('training time:', time.time() - start_time)


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    utils.make_manager()
    n_job = utils.job_increment()
    config.out_dir = os.path.join(config.out_dir, '{:06}'.format(n_job))
    os.makedirs(config.out_dir)
    print('Job number: {:04d}'.format(n_job))

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    train(config)
