#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import time
import datetime
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader
from imageio import imsave
from datasets import AIADataset
from utils.him import SummaryHelper
from models import create_model
from utils.him import calculate_mean_loss, zscore2, imnorm, fitswrite

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='conditional aia generation')
    parser.add_argument('--isTrain', default=True, help='training model or test')
    parser.add_argument('--gpu_num', '-g', type=int, default=0, help='Num. of GPUs')
    parser.add_argument('--train_path', default='../datasets/AIA_CG/train')
    parser.add_argument('--valid_path', default='../datasets/AIA_CG/valid')
    parser.add_argument('--output_root', type=str, default='../outputs/AIA_CG/')
    parser.add_argument('--checkpoints_dir', type=str, default='../outputs/AIA_CG/checkpoints/',
                        help='which epoch to load?')
    parser.add_argument('--samples_dir', type=str, default='../outputs/AIA_CG/samples/', help='which epoch to load?')

    parser.add_argument('--epoch_num', default=500, type=int, help='# total epoch num')
    parser.add_argument('--num_workers', default=0, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--resume', default=False, help='continue training: True or False')
    parser.add_argument('--resume_count', type=int, default=2, help='when resume,from which count to epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--save_valid_img', default=True, help='save image when run valid datasets?')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lambda_L1', type=float, default=50.0, help='')
    parser.add_argument('--model', type=str, default='cgan', help='chooses which model to use. [gan | cnn]')

    parser.add_argument('--save_loss_freq', type=int, default=1, help='frequency of saving the loss results')
    parser.add_argument('--save_model_freq', type=int, default=1, help='frequency of saving checkpoints')
    parser.add_argument('--lr_policy', type=str, default='cosine',
                        help='learning rate policy. [linear | step | plateau | cosine]')

    args = parser.parse_args()
    args.device = torch.device("cuda" if args.gpu_num > 0 else "cpu")

    # mkdirs
    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)
    if not os.path.exists(args.checkpoints_dir):
        os.mkdir(args.checkpoints_dir)
    if not os.path.exists(args.samples_dir):
        os.mkdir(args.samples_dir)

    print('total_epoch_num:', args.epoch_num)
    train_summary = SummaryHelper(save_path=os.path.join(args.output_root, 'log', 'train'),
                                  comment='models', flush_secs=20)
    valid_summary = SummaryHelper(save_path=os.path.join(args.output_root, 'log', 'test'),
                                  comment='models', flush_secs=20)

    # data loaders
    train_data = AIADataset(args.train_path)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    valid_data = AIADataset(args.valid_path)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2,
                              pin_memory=True, drop_last=False)
    train_step = len(train_loader)
    valid_step = len(valid_loader)

    # Fixed random parameters
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # start or restart from checkpoints
    model = create_model(args)
    epoch_start = 1
    if args.resume:
        model.load_networks(args.resume_count)
        epoch_start = args.resume_count + 1
    model.setup(args)

    # training process
    for epoch in range(epoch_start, args.epoch_num + 1):
        epoch_start_time = time.time()
        # **************************    forward    ************************** #
        train_loss_list = []
        for t_i, t_data in enumerate(train_loader):
            model.set_input(t_data)
            model.optimize_parameters()
            train_loss_per = model.get_current_losses()
            train_loss_list.append(train_loss_per)
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'epoch: ' + str(epoch), 'itr: ' + str(t_i),
                  'loss: ', train_loss_per)

        # **************************    save loss    ************************** #
        if epoch % args.save_loss_freq == 0:
            # mean train loss
            train_mean_loss = calculate_mean_loss(train_loss_list)
            train_summary.add_summary(train_mean_loss, global_step=epoch)
            # mean valid loss
            model.eval()
            valid_outputs_list, valid_loss_list = [], []
            for v_i, v_data in enumerate(valid_loader):
                model.set_input(v_data)
                model.test()
                valid_outputs_list.append(model.get_current_np_outputs())
                valid_loss_list.append(model.get_current_losses())
            model.train()
            valid_mean_loss = calculate_mean_loss(valid_loss_list)
            valid_summary.add_summary(valid_mean_loss, global_step=epoch)
            print('mean loss', 'train:', train_mean_loss, 'valid', valid_mean_loss)

            # save sample to file
            valid_outputs_list = np.concatenate(valid_outputs_list, axis=0)
            for i in range(len(valid_outputs_list)):
                saving_img = np.hstack([zscore2(_) for _ in valid_outputs_list[i]])
                saving_img = np.array(imnorm(saving_img) * 255.0, dtype=np.uint8)
                fitswrite('%s/epoch_%d__%d.fits' % (args.samples_dir, epoch, i), valid_outputs_list[i])
                imsave('%s/epoch_%d__%d.jpg' % (args.samples_dir, epoch, i), saving_img)

        # **************************    save model    ************************** #
        if epoch % args.save_model_freq == 0:
            model.save_networks(epoch)

        model.update_learning_rate(epoch)
