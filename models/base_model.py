import os
from abc import ABC, abstractmethod
from collections import OrderedDict
import torch.optim as optim
import torch
from models import net_utils


class BaseModel(ABC):

    def __init__(self, config):
        self.config = config
        self.isTrain = config.isTrain
        self.device = config.device
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.loss_stack = OrderedDict()
        self.metric = 0  # used for learning rate policy 'plateau'

    # 设置输入数据
    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self, opt):
        """
        create schedulers
        """
        if self.isTrain:
            self.schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch_num, eta_min=1e-6) for
                               optimizer in self.optimizers]

            if opt.resume:
                for per_scheduler in self.schedulers:
                    per_scheduler.step(opt.resume_count)
                print('re-adjust learning rate')

    @abstractmethod
    def test(self):
        pass

    def get_lr(self):
        lr = self.optimizers[0].param_groups[0]['lr']
        return lr

    def update_learning_rate(self, epoch):
        """Update learning rates for all the models; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()
        # lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)

    # 返回输出结果
    def get_current_np_outputs(self):
        pass

    # 返回 loss names
    def get_loss_names(self):
        return self.loss_names

    # 返回最近的loss值
    def get_current_losses(self):
        loss_dict = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                loss_dict[name] = float(getattr(self, 'loss_' + name))
        return loss_dict

    # ****************************  save、load、print models *****************************#

    def save_networks(self, epoch):
        """Save all the models to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = 'epoch_%d_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.config.checkpoints_dir, save_filename)
                net = getattr(self, 'net' + name)

                torch.save(net.state_dict(), save_path)
        print('save epoch %d models to file !' % epoch)

    def load_networks(self, epoch):
        """Load all the models from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = 'epoch_%d_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.config.checkpoints_dir, load_filename)
                if not os.path.exists(load_path):
                    continue
                net = getattr(self, 'net' + name)
                print('loading the models from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def print_networks(self):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the models to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of models
            requires_grad (bool)  -- whether the models require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
