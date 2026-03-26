import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention module"""
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        mid_channels = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels),
            nn.Sigmoid()
        )
        for m in self.excitation:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class ResidualStack(nn.Module):
    """
    Residual module: 1x1 linear projection + 2 residual units + MaxPooling
    Adapted for image input, pooling is performed on both H and W dimensions simultaneously
    """
    def __init__(self, in_channels, filter_num, kernel_size=(3, 3), pool_size=(2, 2)):
        super(ResidualStack, self).__init__()
        # 1x1 Conv linear projection
        self.conv1 = nn.Conv2d(in_channels, filter_num, kernel_size=1)

        # Residual unit 1
        self.conv2 = nn.Conv2d(filter_num, filter_num, kernel_size=kernel_size, padding='same')
        self.conv3 = nn.Conv2d(filter_num, filter_num, kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(filter_num)

        # Residual unit 2
        self.conv4 = nn.Conv2d(filter_num, filter_num, kernel_size=kernel_size, padding='same')
        self.conv5 = nn.Conv2d(filter_num, filter_num, kernel_size=kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(filter_num)

        # Max pooling (downsampling on both H and W dimensions)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

        # Weight initialization
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)

        # Residual unit 1
        shortcut = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(x + shortcut)
        x = self.bn1(x)

        # Residual unit 2
        shortcut = x
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = F.relu(x + shortcut)
        x = self.bn2(x)

        x = self.pool(x)
        return x


class FPNBlock(nn.Module):
    """FPN upsampling module: 1x1 convolution + 2D upsampling"""
    def __init__(self, in_channels, out_channels, up_size=(2, 2)):
        super(FPNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up = nn.Upsample(scale_factor=up_size, mode='nearest')

    def forward(self, x):
        return self.up(self.conv(x))


class PANBlock(nn.Module):
    """PAN path enhancement module, selects different processing paths based on upsample parameter"""
    def __init__(self, in_channels, filter_num, kernel_size=(3, 3),
                 pool_size=(2, 2), strides=(2, 2), upsample=True):
        super(PANBlock, self).__init__()
        self.upsample = upsample

        if upsample:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, filter_num, kernel_size=kernel_size, padding='same'),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(filter_num),
                nn.MaxPool2d(kernel_size=pool_size, stride=strides),
            )
        else:
            self.layers = nn.Sequential(
                nn.AvgPool2d(kernel_size=1, stride=1),
                nn.Conv2d(in_channels, filter_num, kernel_size=kernel_size, padding='same'),
                nn.ReLU(inplace=True),
                nn.Conv2d(filter_num, filter_num, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(filter_num),
                nn.MaxPool2d(kernel_size=pool_size, stride=strides),
            )

    def forward(self, x):
        return self.layers(x)


class MFFNet(nn.Module):
    """
    in_channels: Number of input image channels (RGB=3, Grayscale=1, Covariance matrix=3, etc.)
    num_classes: Number of classes
    base_filters: Backbone base channel count, default 32
    fpn_channels: FPN/PAN channel count, default 64
    """
    def __init__(self, in_channels=3, num_classes=4, base_filters=32, fpn_channels=64):
        super(MFFNet, self).__init__()
        self.num_classes = num_classes
        bf = base_filters   # Backbone channel count
        fc = fpn_channels   # FPN/PAN channel count

        # Backbone: 4-stage residual stacks + SE attention
        # Each stage output: H/2^k, W/2^k (k=1,2,3,4)
        self.res_stack0 = ResidualStack(in_channels, bf, kernel_size=(3, 3), pool_size=(2, 2))
        self.se0 = SEBlock(bf, reduction=4)

        self.res_stack1 = ResidualStack(bf, bf * 2, kernel_size=(3, 3), pool_size=(2, 2))
        self.se1 = SEBlock(bf * 2, reduction=4)

        self.res_stack2 = ResidualStack(bf * 2, bf * 4, kernel_size=(3, 3), pool_size=(2, 2))
        self.se2 = SEBlock(bf * 4, reduction=4)

        self.res_stack3 = ResidualStack(bf * 4, bf * 8, kernel_size=(3, 3), pool_size=(2, 2))
        self.se3 = SEBlock(bf * 8, reduction=4)

        # Channel projection: unify all backbone stage outputs to fpn_channels
        self.proj_r1 = nn.Conv2d(bf, fc, kernel_size=1)       # Stage 1: bf   -> fc
        self.proj_r2 = nn.Conv2d(bf * 2, fc, kernel_size=1)   # Stage 2: bf*2 -> fc
        self.proj_r3 = nn.Conv2d(bf * 4, fc, kernel_size=1)   # Stage 3: bf*4 -> fc
        self.proj_y  = nn.Conv2d(bf * 8, fc, kernel_size=1)   # Stage 4: bf*8 -> fc

        # FPN top-down path
        # Each stage: 1x1 convolution + 2x upsampling, then add with shallow features
        self.fpn_p5 = FPNBlock(bf * 8, fc, up_size=(2, 2))
        self.fpn_p5_conv = nn.Conv2d(fc, fc, kernel_size=3, padding=1)

        self.fpn_p4 = FPNBlock(fc, fc, up_size=(2, 2))
        self.fpn_p4_conv = nn.Conv2d(fc, fc, kernel_size=3, padding=1)

        self.fpn_p3 = FPNBlock(fc, fc, up_size=(2, 2))
        self.fpn_p3_conv = nn.Conv2d(fc, fc, kernel_size=3, padding=1)

        self.fpn_p2 = FPNBlock(fc, fc, up_size=(2, 2))
        self.fpn_p2_conv = nn.Conv2d(fc, fc, kernel_size=3, padding=1)

        # PAN bottom-up path enhancement
        self.pan1 = PANBlock(fc, fc, kernel_size=(3, 3), pool_size=(1, 1), strides=(1, 1), upsample=True)
        self.pan2 = PANBlock(fc, fc, kernel_size=(3, 3), pool_size=(2, 2), strides=(2, 2), upsample=True)
        self.pan3 = PANBlock(fc, fc, kernel_size=(3, 3), pool_size=(2, 2), strides=(2, 2), upsample=True)
        self.pan4 = PANBlock(fc, fc, kernel_size=(3, 3), pool_size=(2, 2), strides=(2, 2), upsample=False)
        self.pan5 = PANBlock(fc, fc, kernel_size=(3, 3), pool_size=(2, 2), strides=(2, 2), upsample=False)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Adaptive global pooling, outputs (B, fc, 1, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(fc, num_classes)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, x):
        # Backbone
        x = self.se0(self.res_stack0(x))    # (B, bf,   H/2,  W/2)
        xm_r1 = x

        x = self.se1(self.res_stack1(x))    # (B, bf*2, H/4,  W/4)
        xm_r2 = x

        x = self.se2(self.res_stack2(x))    # (B, bf*4, H/8,  W/8)
        xm_r3 = x

        x = self.se3(self.res_stack3(x))    # (B, bf*8, H/16, W/16)
        xm_y = x

        # FPN top-down
        x = self.fpn_p5(x)                  # (B, fc, H/8,  W/8)
        xm_conv1 = self.fpn_p5_conv(x)      # Save side output
        x = x + self.proj_r3(xm_r3)         # Fuse with stage 3 features

        x = self.fpn_p4(x)                  # (B, fc, H/4,  W/4)
        xm_conv2 = self.fpn_p4_conv(x)
        x = x + self.proj_r2(xm_r2)

        x = self.fpn_p3(x)                  # (B, fc, H/2,  W/2)
        xm_conv3 = self.fpn_p3_conv(x)
        x = x + self.proj_r1(xm_r1)

        x = self.fpn_p2(x)                  # (B, fc, H,    W)
        xm_conv4 = self.fpn_p2_conv(x)

        # PAN bottom-up
        x = self.pan1(x) + xm_conv4         # (B, fc, H,    W)
        x = self.pan2(x) + xm_conv3         # (B, fc, H/2,  W/2)
        x = self.pan3(x) + xm_conv2         # (B, fc, H/4,  W/4)
        x = self.pan4(x) + xm_conv1         # (B, fc, H/8,  W/8)
        x = self.pan5(x) + self.proj_y(xm_y)  # (B, fc, H/16, W/16)

        # Classification head
        x = self.global_pool(x)             # (B, fc, 1, 1) Adaptive pooling, compatible with any resolution
        x = x.flatten(1)                    # (B, fc)
        x = self.dropout(x)
        x = self.classifier(x)              # (B, num_classes)
        return x