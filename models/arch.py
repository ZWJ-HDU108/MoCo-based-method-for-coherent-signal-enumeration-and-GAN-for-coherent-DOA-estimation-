import torch
import torch.nn as nn
from models.arch_util import LayerNorm2d


# Gating mechanism module
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2  # Number of channels halves


class Branch(nn.Module):
    '''
    Branch that lasts only the dilated convolutions
    '''

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation)  # the dconv
        )

    def forward(self, input):
        return self.branch(input)


class EBlock(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()

        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True,
                                    dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw, channel count unchanged, size unchanged
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)  # Channel count doubles, size unchanged

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.convd = nn.Conv2d(in_channels=DW_Expand * c, out_channels=c, kernel_size=1, padding=0, stride=1)
    def forward(self, input):
        y = input  # shape: [1, 3, 256, 256] -> [1, 3, 256, 256]
        x = self.norm1(input)  # shape: [1, 3, 256, 256] -> [1, 3, 256, 256]
        x = self.conv1(self.extra_conv(x))  # shape: [1, 3, 256, 256] -> [1, 6, 256, 256]
        z = 0
        for branch in self.branches:
            z += branch(x)  # shape: [1, 6, 256, 256] -> [1, 6, 256, 256]

        # z = self.sg1(z)  # shape: [1, 6, 256, 256] -> [1, 3, 256, 256]
        z = self.convd(z)  # For ablation study, delete this line
        x = self.sca(z) * z  # shape: [1, 3, 256, 256] -> [1, 3, 256, 256]
        x = self.conv3(x)  # shape: [1, 3, 256, 256] -> [1, 3, 256, 256]
        y = input + self.beta * x  # shape: [1, 3, 256, 256] -> [1, 3, 256, 256]

        x = self.norm2(y)  # shape: [1, 3, 256, 256] -> [1, 3, 256, 256]
        return x  # output size: [1, 3, 256, 256]


class DBlock(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations=[1],
                 extra_depth_wise=False):  # c: image channel, usually=3
        super().__init__()

        self.dw_channel = DW_Expand * c  # =6

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1) # Channel count doubles, size unchanged
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c,
                                    bias=True, dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw, channel count unchanged, size unchanged
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand=1, dilation=dilation)) # Three dilated convolutions with different scales

        assert len(dilations) == len(self.branches)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1, groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.convd = nn.Conv2d(in_channels=DW_Expand * c, out_channels=c, kernel_size=1, padding=0, stride=1)
    def forward(self, input):

        x = self.norm1(input)  # shape: [1, 3, 16, 16] -> [1, 3, 16, 16]
        x = self.extra_conv(self.conv1(x))  # shape: [1, 3, 16, 16] -> [1, 6, 16, 16]
        z = 0
        for branch in self.branches:
            z += branch(x) # After passing through three convolution branches and adding them  shape: [1, 6, 16, 16] -> [1, 6, 16, 16]

        # z = self.sg1(z)  # shape: [1, 6, 16, 16] -> [1, 3, 16, 16]
        z = self.convd(z)  # For ablation study, delete this line
        x = self.sca(z) * z  # shape: [1, 3, 16, 16] -> [1, 3, 16, 16]
        x = self.conv3(x)  # shape: [1, 3, 16, 16] -> [1, 3, 16, 16]
        y = input + self.beta * x  # shape: [1, 3, 16, 16] -> [1, 3, 16, 16]
        # second step
        x = self.conv4(self.norm2(y))  # shape: [1, 3, 16, 16] -> [1, 6, 16, 16]
        x = self.sg2(x)  # shape: [1, 6, 16, 16] -> [1, 3, 16, 16]
        x = self.conv5(x)  # shape: [1, 3, 16, 16] -> [1, 3, 16, 16]
        x = y + x * self.gamma  # shape: [1, 3, 16, 16] -> [1, 3, 16, 16]

        return x  # output size: [1, 3, 16, 16]


if __name__ == '__main__':
    img_channel = 3
    width = 32

    dilations = [1, 4, 9]
    extra_depth_wise = True

    net1 = EBlock(c=img_channel,
                 dilations=dilations,
                 extra_depth_wise=extra_depth_wise)

    net2 = DBlock(c=img_channel,
                  dilations=dilations,
                  extra_depth_wise=extra_depth_wise)

    input = torch.randn(1, 3, 16, 16)

    from thop import profile

    flops, params = profile(net1, inputs=(input, ))
    print(f"FLOPs: {flops / 1e3:.2f} K")
    print(f"Params: {params / 1e3:.2f} K")