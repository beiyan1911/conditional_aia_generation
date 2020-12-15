from models.ops import *
import torch


class CNN(nn.Module):
    def __init__(self, in_dims=3, out_dims=1, n_feats=32, layers_num=4):
        super(CNN, self).__init__()
        kernel_size = 3
        layers = []
        self.conv_in = nn.Sequential(nn.Conv2d(in_dims, n_feats, kernel_size, stride=1, padding=1, bias=False),
                                     nn.ReLU())
        for i in range(layers_num):
            layers += [nn.Conv2d(n_feats, n_feats, kernel_size, stride=1, padding=1, bias=False), nn.ReLU()]
        self.body = nn.Sequential(*layers)

        self.conv_out = nn.Conv2d(n_feats, out_dims, kernel_size, stride=1, padding=1, bias=False)

    def forward(self, x, condition):
        out = self.conv_in(x)
        out = self.body(out)
        out = self.conv_out(out)
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            DoubleConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels+3, in_channels // 2, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2,condition):
        W = x1.size(2)
        c_pad = condition.unsqueeze(2).unsqueeze(3).repeat([1,1,W,W])
        x = torch.cat([x1, c_pad], dim=1)
        x1 = self.up(x)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# class UNet(nn.Module):
#     def __init__(self, in_dims, out_dims, n_feat=64):
#         super(UNet, self).__init__()
#
#         self.inc = DoubleConv(in_dims, n_feat)
#         self.down1 = Down(n_feat, n_feat * 2)
#         self.down2 = Down(n_feat * 2, n_feat * 4)
#         self.down3 = Down(n_feat * 4, n_feat * 8)
#         self.down4 = Down(n_feat * 8, n_feat * 16)
#         self.up1 = Up(n_feat * 16, n_feat * 8)
#         self.up2 = Up(n_feat * 8, n_feat * 4)
#         self.up3 = Up(n_feat * 4, n_feat * 2)
#         self.up4 = Up(n_feat * 2, n_feat)
#         self.outc = OutConv(n_feat, out_dims)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         out = self.outc(x)
#         return out


class UNet(nn.Module):
    def __init__(self, in_dims, out_dims, n_feat=4):
        super(UNet, self).__init__()

        self.inc = DoubleConv(in_dims, n_feat)
        self.down1 = Down(n_feat, n_feat * 2)
        self.down2 = Down(n_feat * 2, n_feat * 4)
        self.up1 = Up(n_feat * 4, n_feat * 2)
        self.up2 = Up(n_feat * 2, n_feat)
        self.outc = OutConv(n_feat, out_dims)

    def forward(self, x, condition):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2,condition)
        x = self.up2(x, x1,condition)
        out = self.outc(x)
        return out


class Resnet(nn.Module):
    def __init__(self, in_dims=1, out_dims=1, n_feats=32, layers_num=30):
        super(Resnet, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1

        self.conv_in = nn.Sequential(nn.Conv2d(in_dims, n_feats, kernel_size, stride, padding, bias=True),
                                     nn.ReLU())
        layers = []
        for i in range(layers_num):
            layers += [ResBlock(n_feats, kernel_size, stride, padding, bias=True)]
        self.body = nn.Sequential(*layers)
        self.conv_out = nn.Conv2d(n_feats, out_dims, kernel_size, stride, padding, bias=True)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.body(out)
        out = self.conv_out(out)
        return out


class FisaModel(nn.Module):
    def __init__(self, in_dims=1, out_dims=1, n_feats=24, layers_num=15):
        super(FisaModel, self).__init__()

    def forward(self, x):
        return x
