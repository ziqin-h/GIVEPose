from network.ops_dcnv3.modules.dcnv3 import DCNv3_pytorch,DCNv3
import torch.nn as nn



class DCNv3_PyTorch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=4, dilation=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=padding)
        self.dcnv3 = DCNv3_pytorch(out_channels, kernel_size=kernel_size, stride=stride, group=groups,
                                   dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.dcnv3(x)
        x = x.permute(0, 3, 1, 2)
        # x = self.gelu(self.bn(x))
        return x

class DCNv3_C(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=4, dilation=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dcnv3 = DCNv3(out_channels, kernel_size=kernel_size, stride=stride, group=groups,
                                   dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.dcnv3(x)
        x = x.permute(0, 3, 1, 2)
        # x = self.gelu(self.bn(x))
        return x