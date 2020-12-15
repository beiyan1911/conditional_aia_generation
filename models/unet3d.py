import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, batchNorm=True, affine=False):
        super().__init__()

        self.deconv1 = nn.ConvTranspose3d(in_channels + 1, in_channels // 2, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                                          padding=(1, 1, 1), output_padding=(0, 1, 1),
                                          bias=bias)

        if batchNorm:
            self.bn1 = nn.BatchNorm3d(in_channels // 2, affine=affine)
        else:
            self.bn1 = Identity()
        self.relu = nn.ReLU()

        self.deconv2 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                          padding=(1, 1, 1), bias=bias)
        if batchNorm:
            self.bn2 = nn.BatchNorm3d(out_channels, affine=affine)

    def forward(self, x1, x2, condition):
        w, h = x1.size(3), x1.size(4)
        c = condition.unsqueeze(1).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, w, h)
        x_c_cat = torch.cat([x1, c], dim=1)
        res = self.deconv1(x_c_cat)
        res = self.bn1(res)
        res = self.relu(res)

        res = torch.cat([res, x2], dim=1)
        res = self.deconv2(res)
        res = self.bn1(res)
        res = self.relu(res)
        return res


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, batchnorm=True, is_out_layer=False, affine=False):
        super().__init__()
        layers = []
        if is_out_layer:  # 最外层不下采样,非最外层下采样
            layers += [nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                                 bias=bias)]
        else:
            layers += [nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                                 bias=bias)]
        if batchnorm:
            layers += [nn.BatchNorm3d(out_channels, affine=affine)]
        layers += [nn.ReLU()]
        layers += [nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                             bias=bias)]
        if batchnorm:
            layers += [nn.BatchNorm3d(out_channels, affine=affine)]
        layers += [nn.ReLU()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNet3D(nn.Module):
    def __init__(self, out_channels=1, n_feat=8, bias=True, batchNorm=True, affine=False):
        self.in_channel = 1
        super(UNet3D, self).__init__()
        self.cin = Encoder(self.in_channel, n_feat, bias=bias, batchnorm=batchNorm, affine=affine,
                           is_out_layer=True)
        self.ec1 = Encoder(n_feat * 1, n_feat * 2, bias=bias, batchnorm=batchNorm, affine=affine)
        self.ec2 = Encoder(n_feat * 2, n_feat * 4, bias=bias, batchnorm=batchNorm, affine=affine)
        self.ec3 = Encoder(n_feat * 4, n_feat * 8, bias=bias, batchnorm=batchNorm, affine=affine)

        self.dc3 = Decoder(n_feat * 8, n_feat * 4, bias=bias, batchNorm=batchNorm, affine=affine)
        self.dc2 = Decoder(n_feat * 4, n_feat * 2, bias=bias, batchNorm=batchNorm, affine=affine)
        self.dc1 = Decoder(n_feat * 2, n_feat * 1, bias=bias, batchNorm=batchNorm, affine=affine)

        self.cout = nn.Sequential(
            nn.Conv3d(n_feat, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                      bias=bias))

    def forward(self, x, condition):
        x = x.unsqueeze(1)
        e1 = self.cin(x)
        e2 = self.ec1(e1)
        e3 = self.ec2(e2)
        e4 = self.ec3(e3)
        o3 = self.dc3(e4, e3, condition)
        o2 = self.dc2(o3, e2, condition)
        e0 = self.dc1(o2, e1, condition)
        res = self.cout(e0)
        return res.squeeze(2)


if __name__ == '__main__':
    x = torch.randn([1, 3, 512, 512], dtype=torch.float32)
    condition = torch.randn([1, 3], dtype=torch.float32)
    model = UNet3D(1, 1)
    y = model(x, condition)
    print()
