from torch import nn


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, stride, padding, bias=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out
