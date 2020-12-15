import numpy as np
import torch
from models.simple_net import CNN, UNet
from models.networks import NLayerDiscriminator
from models.net_utils import TVLoss, init_weights
from utils.him import tensor2np
from .base_model import BaseModel


class CGANModel(BaseModel):

    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.loss_names = ['G_sup']  # sub 表示像素损失
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = UNet(3, 1).to(self.device)

        init_weights(self.netG)

        self.const_conditions = [np.tile(np.array([[1, 0, 0]], dtype=np.float32), [self.config.batch_size, 1]),
                                 np.tile(np.array([[0, 1, 0]], dtype=np.float32), [self.config.batch_size, 1]),
                                 np.tile(np.array([[0, 0, 1]], dtype=np.float32), [self.config.batch_size, 1])
                                 ]

        if self.isTrain:
            # L1 = torch.nn.L1Loss() , L2 = torch.nn.MSELoss()

            self.netD = NLayerDiscriminator(input_nc=7, ndf=8, n_layers=3).to(self.device)
            init_weights(self.netD)

            # self.criterionGAN = torch.nn.MSELoss()
            self.criterionGAN = torch.nn.L1Loss()
            self.True_ = torch.tensor(1.0).to(self.device)
            self.False_ = torch.tensor(0.0).to(self.device)

            self.supLoss = torch.nn.MSELoss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, label_idx=0, keep_in=False):
        """
        设置输入数据
        :param input:
        :param label_idx: label index
        :param keep_in: 是否保持输入数据不变，更新label
        :return:
        """

        if not keep_in:
            self.real_A = input['inputs'].to(self.device)

        if 'outputs' in input.keys():
            self.real_B = input['outputs'][:, label_idx:label_idx + 1].to(self.device)

        self.condition = torch.from_numpy(self.const_conditions[label_idx]).to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A, self.condition)

    def test(self):
        with torch.no_grad():
            self.forward()
            self.loss_G_sup = self.supLoss(self.fake_B, self.real_B) * self.config.lambda_L1

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # ************************    update D    ***********************
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero

        # Fake
        W = self.fake_B.size(2)
        c_pad = self.condition.unsqueeze(2).unsqueeze(3).repeat(1, 1, W, W)
        fake_AB = torch.cat((self.real_A, self.fake_B.detach(), c_pad), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, self.False_.expand_as(pred_fake))

        self.loss_D_fake.backward()

        # Real
        real_AB = torch.cat((self.real_A, self.real_B, c_pad), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, self.True_.expand_as(pred_real))

        self.loss_D_real.backward()

        # combine
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.optimizer_D.step()  # update D's weights
        # ************************    update G    ***********************
        # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero

        fake_AB = torch.cat((self.real_A, self.fake_B.detach(), c_pad), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, self.True_.expand_as(pred_fake))

        self.loss_G_GAN.backward()

        # Second, G(A) = B
        self.loss_G_sup = self.supLoss(self.fake_B, self.real_B) * self.config.lambda_L1
        self.loss_G_sup.backward()

        # combine
        self.loss_G = self.loss_G_GAN + self.loss_G_sup

        self.optimizer_G.step()

    def get_current_np_outputs(self, only_out=False):
        """
        return 4 dims data. [N,C,W,H]
        """

        # consist of inputs and outputs
        if only_out:
            res = tensor2np(self.fake_B)
        else:
            label = tensor2np(self.real_B)
            predict = tensor2np(self.fake_B)
            diff = np.abs(label - predict)
            inputs = tensor2np(self.real_A[:, 0:1])
            res = np.concatenate([inputs, label, predict, diff], axis=1)

        return res
