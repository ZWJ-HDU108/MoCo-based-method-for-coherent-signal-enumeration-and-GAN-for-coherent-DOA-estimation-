import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.arch import EBlock, DBlock
    from models.arch_util import CustomSequential
except:
    raise ImportError('Please upgrade your network!')


class GAN_generator(nn.Module):
    def __init__(self,
                 img_channel=3,
                 width=32,
                 middle_blk_num_enc=1,
                 middle_blk_num_dec=1,
                 enc_blk_nums=[1, 2, 2],
                 dec_blk_nums=[2, 2, 1],
                 dilations = [1, 4, 9],
                 extra_depth_wise = True):
        super(GAN_generator, self).__init__()

        self.intro = nn.Conv2d(in_channels=img_channel,  # =3
                               out_channels=width,  # =32
                               kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width,  # =32
                                out_channels=img_channel,  # =3
                                kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks_enc = nn.ModuleList()
        self.middle_blks_dec = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2  # Channel count doubles after each downsampling

        self.middle_blks_enc = nn.Sequential(
                *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
            )
        self.middle_blks_dec = nn.Sequential(
                *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in
                  range(middle_blk_num_dec)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)  # Pixel shuffle operation for efficient upsampling
                )
            )
            chan = chan // 2  # Channel count halves after each upsampling
            self.decoders.append(
                nn.Sequential(
                    *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)  # Used for check_image_size function

    def forward(self, input):
        _, _, H, W = input.shape

        input = self.check_image_size(input)
        x = self.intro(input)
        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # Apply the encoder transforms
        x_light = self.middle_blks_enc(x)

        # Apply the decoder transforms
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = x + input
        out = x[:, :, :H, :W]  # Recover the original size of the image

        return out

    # When input data dimensions are not divisible by padder_size, padding is required
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x


class GAN_discriminator(nn.Module):
    def __init__(self):
        super(GAN_discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        out = self.discriminator(input)
        return out


if __name__ == '__main__':
    import time
    img_channel = 3
    width = 32

    enc_blks = [1, 2, 2]
    middle_blk_num_enc = 1
    middle_blk_num_dec = 1
    dec_blks = [2, 2, 1]
    residual_layers = None
    dilations = [1, 4, 9]
    extra_depth_wise = True

    generator = GAN_generator(img_channel=img_channel,
                 width=width,
                 middle_blk_num_enc=middle_blk_num_enc,
                 middle_blk_num_dec=middle_blk_num_dec,
                 enc_blk_nums=enc_blks,
                 dec_blk_nums=dec_blks,
                 dilations=dilations,
                 extra_depth_wise=extra_depth_wise)

    discriminator = GAN_discriminator()
    start = time.time()

    input = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        output = generator(input)

    end = time.time()
    print(end - start)

    # from thop import profile
    #
    # flops, params = profile(discriminator, inputs=(input,))
    # print(f"FLOPs: {flops / 1e6:.2f} M")
    # print(f"Params: {params / 1e6:.2f} M")