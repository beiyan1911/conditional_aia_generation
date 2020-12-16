import torch
import torch.nn as nn
from models.networks import ResnetBlock, DoubleConv
import models.net_utils as utils


class TrunkBranch(nn.Module):
    def __init__(self, n_feat, kernel_size=3, stride=1, padding=1, use_bias=False, norm=False,
                 norm_layer=nn.InstanceNorm2d):
        super(TrunkBranch, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body += [ResnetBlock(n_feat, kernel_size, stride, padding, use_bias, norm, norm_layer)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x)


class MaskBranch(nn.Module):
    def __init__(self, n_feat, kernel_size=3, stride=1, padding=1, use_bias=False, norm=False,
                 norm_layer=nn.InstanceNorm2d):
        super(MaskBranch, self).__init__()

        # ********* 原有 attention
        self.rb1 = ResnetBlock(n_feat, kernel_size, stride, padding, use_bias, norm_layer)
        # self.down = nn.Sequential(nn.ReflectionPad2d(1),
        #                           nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=2),
        #                           # nn.ReflectionPad2d(1),
        #                           # nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=2)
        #                           )
        # self.rb2 = nn.Sequential(ResnetBlock(n_feat, kernel_size, stride, padding, use_bias, norm, norm_layer),
        #                          ResnetBlock(n_feat, kernel_size, stride, padding, use_bias, norm, norm_layer))
        # self.up = nn.Sequential(nn.ConvTranspose2d(n_feat, n_feat, 3, stride=2, padding=1, output_padding=1),
        #                         # nn.ConvTranspose2d(n_feat, n_feat, 3, stride=2, padding=1, output_padding=1)
        #                         )
        # self.rb3 = ResnetBlock(n_feat, kernel_size, stride, padding, use_bias, norm, norm_layer)
        # self.conv1x1 = nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=use_bias)
        # self.sigmoid = nn.Sigmoid()
        #
        self.model = nn.Sequential(
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, bias=use_bias),
            ResnetBlock(n_feat, kernel_size, stride, padding, use_bias, norm_layer),
            ResnetBlock(n_feat, kernel_size, stride, padding, use_bias, norm_layer),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=use_bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # ********* 原有 attention
        # x_rb1 = self.rb1(x)
        # x_dowm = self.down(x_rb1)
        # x_rb2 = self.rb2(x_dowm)
        # x_up = self.up(x_rb2)
        # x_prerb3 = x_rb1 + x_up
        # x_rb3 = self.rb3(x_prerb3)
        # x_1x1 = self.conv1x1(x_rb3)
        # mx = self.sigmoid(x_1x1)
        # return mx
        return self.model(x)


class ResAttModule(nn.Module):
    def __init__(self, n_feat, kernel_size=3, stride=1, padding=1, use_bias=False, norm=False,
                 norm_layer=nn.InstanceNorm2d):
        super(ResAttModule, self).__init__()
        self.res_block = ResnetBlock(n_feat, kernel_size, stride, padding, use_bias, norm_layer)
        self.trunk = TrunkBranch(n_feat, kernel_size, stride, padding, use_bias, norm, norm_layer)
        self.mask = MaskBranch(n_feat, kernel_size, stride, padding, use_bias, norm, norm_layer)
        self.end = nn.Sequential(ResnetBlock(n_feat, kernel_size, stride, padding, use_bias, norm, norm_layer))

    def forward(self, x):
        RA_RB1_x = self.res_block(x)
        tx = self.trunk(RA_RB1_x)
        out = self.mask(RA_RB1_x)
        out = tx * out
        out = out + RA_RB1_x
        out = self.end(out)
        return out


class FeatureModule(nn.Module):

    def __init__(self, in_dims, nf, use_bias=False, norm=False, norm_layer=nn.InstanceNorm2d):
        super(FeatureModule, self).__init__()
        # self.conv_1 = nn.Conv2d(in_dims, onf, 1, 1, bias=False)

        # self.conv_3 = nn.Sequential(nn.ReflectionPad2d(1),
        #                             nn.Conv2d(in_dims, nf, 3, 1, bias=False))
        #
        # self.conv_5 = nn.Sequential(nn.ReflectionPad2d(2),
        #                             nn.Conv2d(in_dims, onf, 5, 1, bias=False))
        #
        # self.conv_7 = nn.Sequential(nn.ReflectionPad2d(3),
        #                             nn.Conv2d(in_dims, onf, 7, 1, bias=False))

        self.conv_double = nn.Sequential(

            DoubleConv(in_dims, nf, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm,
                       norm_layer=norm_layer),
            DoubleConv(nf, nf, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm, norm_layer=norm_layer),
        )
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # feature_1 = self.conv_1(x)
        # feature_3 = self.conv_3(x)
        # feature_5 = self.conv_5(x)
        # features = self.conv_7(x)
        # features = torch.cat([feature_3, feature_5, features], dim=1)
        # del feature_3, feature_5
        # features = self.relu(features)
        return self.conv_double(x)


class MultiAttResnet(nn.Module):

    def __init__(self, in_dims=5, out_dims=1, n_res=6, n_att=2, nf=32, norm_type='instance'):
        super(MultiAttResnet, self).__init__()
        norm_layer = utils.get_norm_layer(norm_type=norm_type)
        norm = True
        use_bias = False
        # ------> 浅层特征 <------ #
        self.base_block = FeatureModule(in_dims, nf=nf, use_bias=use_bias, norm=norm, norm_layer=norm_layer)

        # ------> 注意力特征 <------ #
        self.att_blocks = nn.Sequential(
            # 下采样
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2),
            # 注意力模块
            ResAttModule(nf * 2, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm,
                         norm_layer=norm_layer),
            # 下采样
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=2),
            # 注意力模块
            ResAttModule(nf * 2, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm,
                         norm_layer=norm_layer),
            # 上采样
            nn.ConvTranspose2d(nf * 2, nf * 2, 3, stride=2, padding=1, output_padding=1),
            # 注意力模块
            ResAttModule(nf * 2, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm,
                         norm_layer=norm_layer),
            # 上采样
            nn.ConvTranspose2d(nf * 2, nf, 3, stride=2, padding=1, output_padding=1))

        # ------> 主干特征 <------ #
        # 两次下采样
        self.res_blocks = nn.Sequential(
            # 下采样
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2),
            # 两层残差
            ResnetBlock(nf * 2, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm,
                        norm_layer=norm_layer),
            ResnetBlock(nf * 2, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm,
                        norm_layer=norm_layer),
            # 下采样
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=2),
            # 2层残差
            ResnetBlock(nf * 2, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm,
                        norm_layer=norm_layer),
            ResnetBlock(nf * 2, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm,
                        norm_layer=norm_layer),
            # 上采样
            nn.ConvTranspose2d(nf * 2, nf * 2, 3, stride=2, padding=1, output_padding=1),
            # 两层残差
            ResnetBlock(nf * 2, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm,
                        norm_layer=norm_layer),
            ResnetBlock(nf * 2, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm,
                        norm_layer=norm_layer),
            # 上采样
            nn.ConvTranspose2d(nf * 2, nf, 3, stride=2, padding=1, output_padding=1))

        # ------> 输出层 <------ #
        self.out_blocks = nn.Sequential(

            # DoubleConv(nf * 3, nf, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm,
            #            norm_layer=norm_layer),
            nn.Conv2d(nf * 3, nf, 3, 1, padding=1, bias=use_bias),
            # ResnetBlock(nf, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm, norm_layer=norm_layer),
            # ResnetBlock(nf, kernel_size=3, stride=1, padding=1, use_bias=use_bias, norm=norm, norm_layer=norm_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, out_dims, 3, 1, padding=1, bias=use_bias)
        )

    def forward(self, x):
        base_features = self.base_block(x)
        att_features = self.att_blocks(base_features)

        self.extra_out = att_features[0, :5]

        res_features = self.res_blocks(base_features)

        out = torch.cat([base_features, att_features, res_features], dim=1)
        out = self.out_blocks(out)
        return out

    def get_extra_outputs(self):
        return self.extra_out


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_dims, ndf=32, n_layers=5, norm_type='instance', use_bias=False):
        super(NLayerDiscriminator, self).__init__()
        # norm_layer = utils.get_norm_layer(norm_type=norm_type)
        # spectral_norm

        k = 3
        pad = 0
        layers = [nn.Conv2d(input_dims, ndf, kernel_size=k, stride=2, padding=pad), nn.ReLU(True)]
        nf_mult = 1

        # 下采样
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            layers += [
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=k, stride=2, padding=pad, bias=use_bias)),
                nn.ReLU(True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 4)
        layers += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=k, stride=1, padding=pad, bias=use_bias)),
            nn.ReLU(True)
        ]

        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=k, stride=1, padding=pad)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NAttentionDiscriminator(nn.Module):

    def __init__(self, input_dims, nf=32, n_layers=4, norm_type='instance', use_bias=False):
        super(NAttentionDiscriminator, self).__init__()

        norm_layer = utils.get_norm_layer(norm_type=norm_type)

        layers = [nn.utils.spectral_norm(nn.Conv2d(input_dims, nf, 3, 1, padding=1, bias=False)),
                  # norm_layer(nf),
                  nn.ReLU(True),
                  nn.utils.spectral_norm(nn.Conv2d(nf, nf, 3, 1, padding=1, bias=False)),
                  # norm_layer(nf),
                  nn.ReLU(True)]

        # for i in range(2):
        layers += [ResAttModule(nf, kernel_size=3, stride=1, padding=1, use_bias=False, norm_layer=norm_layer)]

        for n in range(1, n_layers):
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=0, bias=use_bias)),
                # norm_layer(nf),
                # nn.LeakyReLU(0.2, True)
                nn.ReLU(True)
            ]

        layers += [nn.Conv2d(nf, 1, kernel_size=3, stride=2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == '__main__':
    # import torch

    # x = torch.randn(1, 32, 480, 480)
    # model = ResAttModule(32)
    x = torch.randn(1, 5, 512, 512)
    # model = NLayerDiscriminator(6)
    # model = NAttentionDiscriminator(1)
    # model = FeatureModule(5, 16)
    model = MultiAttResnet(in_dims=5)

    out = model(x)
    print(out.shape)

    # x = torch.randn(1, 5, 256, 239)
    # model = nn.ConvTranspose2d(5, 5, 3, stride=2, padding=1, output_padding=1)
    # out = model(x)
    # print(out.shape)
