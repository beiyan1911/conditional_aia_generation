#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader
from imageio import imsave
from datasets import AIADataset
from utils.him import SummaryHelper
from models import create_model
from tqdm import tqdm
from utils.him import calculate_mean_loss, zscore2, imnorm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='conditional aia generation')
    parser.add_argument('--gpu_num', '-g', type=int, default=0, help='Num. of GPUs')
    parser.add_argument('--train_path', default='../datasets/AIA_CG/train')
    parser.add_argument('--valid_path', default='../datasets/AIA_CG/valid')
    parser.add_argument('--output_root', type=str, default='../outputs/AIA_CG/')
    parser.add_argument('--num_workers', default=5, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--checkpoints_dir', type=str, default='../outputs/AIA_CG/checkpoints/',
                        help='which epoch to load?')
    parser.add_argument('--samples_dir', type=str, default='../outputs/AIA_CG/samples/', help='which epoch to load?')
    parser.add_argument('--resume', default=False, help='continue training: True or False')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--resume_count', type=int, default=0, help='when resume,from which count to epoch')
    parser.add_argument('--save_valid_img', default=True, help='save image when run valid datasets?')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--calculate_metric', default=False, help='whether to calculate PSNR and SSIM metrics?')
    parser.add_argument('--model', type=str, default='cgan', help='chooses which model to use. [gan | cnn]')
    parser.add_argument('--save_loss_freq', type=int, default=1, help='frequency of saving the loss results')
    parser.add_argument('--save_model_freq', type=int, default=1,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--lr_policy', type=str, default='plateau',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--save_sample_freq', type=int, default=1,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--lambda_L1', type=float, default=50.0, help='')
    args = parser.parse_args()
    args.isTrain = True

    args.device = torch.device("cuda" if args.gpu_num > 0 else "cpu")
    epoch_num = 100
    test_interval = 5  # 多少epoch跑一遍测试集
    ckpt_interval = 5  # 多少epoch 保存 ckpt

    # 创建目录
    if not os.path.exists(args.checkpoints_dir):
        os.mkdir(args.checkpoints_dir)

    if not os.path.exists(args.samples_dir):
        os.mkdir(args.samples_dir)

    print('total_epoch_num:', epoch_num)
    train_summary = SummaryHelper(save_path=os.path.join(args.output_root, 'log', 'train'),
                                  comment='models', flush_secs=20)
    test_summary = SummaryHelper(save_path=os.path.join(args.output_root, 'log', 'test'),
                                 comment='models', flush_secs=20)

    train_data = AIADataset(args.train_path)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)

    valid_data = AIADataset(args.valid_path)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2,
                              pin_memory=True, drop_last=False)

    train_step = len(train_loader)
    valid_step = len(valid_loader)
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model = create_model(args)

    # ******** restart from checkpoints
    epoch_start = 1
    if args.resume:
        state_dict = torch.load(args.output_root + args.checkpoint, map_location=args.device)
        model.load_state_dict(state_dict)
        epoch_start = args.resume_count
    model.setup(args)  # 启动模型
    for epoch in range(epoch_start, epoch_num + 1):
        epoch_start_time = time.time()

        # **************************    forward    ************************** #
        for i, data in tqdm(enumerate(train_loader), desc='Epoch: [%d/%d]' % (epoch, epoch_num), total=train_step):
            # 生成第一组结果
            model.set_input(data, label_idx=0, keep_in=False)
            model.optimize_parameters()
            loss_1 = model.get_current_losses()

            # 生成第二组结果
            model.set_input(data, label_idx=1, keep_in=True)
            model.optimize_parameters()
            loss_2 = model.get_current_losses()

            # 生成第三组结果
            model.set_input(data, label_idx=2, keep_in=True)
            model.optimize_parameters()
            loss_3 = model.get_current_losses()

        # **************************    save loss    ************************** #
        if epoch % args.save_loss_freq == 0:
            # 计算平均loss
            mean_loss = calculate_mean_loss([loss_1, loss_2, loss_3])

            print('epoch: %d' % epoch, mean_loss)

            # valid loss
            # model.eval()
            # for v_i, v_data in enumerate(valid_loader):
            #     model.set_input(v_data)
            #     model.test()
            # model.train()
            # valid_mean_loss = loss_stack.get_mean(valid_step)
            # valid_Summary.add_summary(valid_mean_loss, global_step=epoch)
        # **************************    save model    ************************** #
        if epoch % args.save_model_freq == 0:
            model.save_networks(epoch)

        # **************************  sample  ************************** #
        valid_outputs = []  # N C W H
        if epoch % args.save_sample_freq == 0:
            # extra sample
            # att_img = model.get_extra_outputs()
            # att_img = np.array((att_img - att_img.min()) / (att_img.max() - att_img.min()) * 255.0, dtype=np.uint8)
            # imsave('%s/epoch_%d_att.jpg' % (opt.sample_dir, epoch), att_img)

            # valid sample
            model.eval()
            for v_i, v_data in enumerate(valid_loader):
                model.set_input(v_data, label_idx=0, keep_in=False)
                model.test()
                valid_outputs.append(model.get_current_np_outputs())

                model.set_input(v_data, label_idx=1, keep_in=True)
                model.test()
                valid_outputs.append(model.get_current_np_outputs())

                model.set_input(v_data, label_idx=1, keep_in=True)
                model.test()
                valid_outputs.append(model.get_current_np_outputs())

            model.train()
            valid_outputs = np.concatenate(valid_outputs, axis=0)

            # save sample to file
            for i in range(len(valid_outputs)):
                saving_img = np.hstack([zscore2(_) for _ in valid_outputs[i]])
                saving_img = np.array(imnorm(saving_img) * 255.0, dtype=np.uint8)
                imsave('%s/epoch_%d__%d.jpg' % (args.samples_dir, epoch, i), saving_img)

        print('\n [Epoch %d / %d] \t Time: %d sec\n' % (epoch, epoch_num, time.time() - epoch_start_time))
        model.update_learning_rate(epoch)
        # torch.cuda.empty_cache()  # 清空 GPU
