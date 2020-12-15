#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import time
import numpy as np
import torch.utils.data
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ski_ssim
from model.simple_net import Resnet
from model.unet import UnetGenerator
from model.unet_standard import UNet
from model.autoencode_unet import AutoEncodeUnet
import csv
from torch import nn
from utils.him import fitswrite

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
parser.add_argument('--num_work', default=0, type=int, help='# threads for loading data')
parser.add_argument('--save_image', default=True, help='where save image?')
parser.add_argument('--checkpoints_dir', type=str, default='')
parser.add_argument('--batch_size', default=1, type=int, help='')

args = parser.parse_args()
from datasets import AIADataset

if __name__ == '__main__':

    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    device = torch.device("cuda" if args.gpu_num > 0 else "cpu")

    # ******** 测试图片目录
    test_root = '../datasets/AIA/test'
    save_root = '../outputs/AIA_autoencode_unet/tests'

    # test 作为验证集
    test_sets = AIADataset(test_root)
    test_loader = DataLoader(dataset=test_sets, num_workers=args.num_work,
                             batch_size=args.batch_size,
                             shuffle=False, pin_memory=True)

    # ******** for demo models
    checkpoints = 'epoch_100.pth'
    args.checkpoints_dir = '../outputs/AIA_autoencode_unet/checkpoints/'
    # models = UNet(6, 6, 32).to(device)
    model = AutoEncodeUnet(6, 6, 32).to(device)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for module in model.children():
        module.train(False)

    state_dict = torch.load(os.path.join(args.checkpoints_dir, checkpoints), map_location=device)
    model.load_state_dict(state_dict)
    print('======> test on checkpoints %s <======' % checkpoints)

    start_time = time.time()
    test_ite = 0
    with torch.no_grad():
        per_start_time = time.time()

        for ti, (input, target, names) in enumerate(test_loader):
            realA = input.to(device)
            realB = target.to(device)
            fakeB = model(realA)
            # save images   normalize=True 时，保存完图片后，不会改变输入tensor
            if args.save_image:
                diff = fakeB - realB
                diff -= diff.min()
                img = torch.cat([realB, fakeB, diff], dim=1).squeeze(0).detach().cpu().numpy()
                save_path = os.path.join(save_root, names[0])
                fitswrite(save_path, img)
                print(names[0])
