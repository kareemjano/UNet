from torch import nn
import torch
import math


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activ_fn = nn.ReLU):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            activ_fn(),
            nn.BatchNorm2d(out_channels),
        )

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class StackedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activ_fn=nn.ReLU):
        super().__init__()

        self.model = nn.Sequential(
            ConvRelu(in_channels, out_channels, kernel_size, padding, activ_fn=activ_fn),
            ConvRelu(out_channels, out_channels, kernel_size, padding, activ_fn=activ_fn),
        )

    def forward(self, x):
        return self.model(x)


class EncoderBlock(nn.Module):
    def __init__(self, input_channels, f_channels, f, padding, activ_fn=nn.ReLU):
        super().__init__()

        self.conv = StackedConv(input_channels, f_channels, f, padding, activ_fn=activ_fn)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x_down, x


class UpBlock(nn.Module):
    def __init__(self, input_channels, f):
        super().__init__()

        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=input_channels, out_channels=int(input_channels/2), kernel_size=f, padding=1),
        )

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.model.apply(weights_init)


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, f_channels, f, padding, activ_fn=nn.ReLU):
        super().__init__()

        self.up = UpBlock(input_channels, f)
        self.conv = StackedConv(input_channels, f_channels, f, padding, activ_fn=activ_fn)

    def forward(self, x, x_pre):
        x_up = self.up(x)
        dx = x_pre.shape[2] - x_up.shape[2]
        dy = x_pre.shape[3] - x_up.shape[3]
        startx, endx = int(dx / 2), -int(dx / 2)
        starty, endy = int(dy / 2), -int(dy / 2)

        x_pre_cropped = x_pre[:,:,startx:endx, starty:endy]
        assert x_pre_cropped.shape == x_up.shape

        x_concat = torch.cat([x_up, x_pre_cropped], dim=1)

        return self.conv(x_concat)


class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        ## Parameters
        activ_fn = nn.ReLU
        input_channels = 3
        f_channels = 64
        f = 3
        padding = 0

        self.input_size = 572
        self.output_size = 388
        # encoder
        self.encoder1 = EncoderBlock(input_channels, f_channels, f, padding, activ_fn=activ_fn)
        self.encoder2 = EncoderBlock(f_channels, f_channels * 2, f, padding, activ_fn=activ_fn)
        self.encoder3 = EncoderBlock(f_channels * 2, f_channels * 4, f, padding, activ_fn=activ_fn)
        self.encoder4 = EncoderBlock(f_channels * 4, f_channels * 8, f, padding, activ_fn=activ_fn)
        self.encoder_out = StackedConv(f_channels * 8, f_channels * 16, f, padding, activ_fn=activ_fn) #1024x28x28

        # decoder
        self.decoder1 = DecoderBlock(f_channels * 16, f_channels * 8, f, padding, activ_fn=activ_fn)
        self.decoder2 = DecoderBlock(f_channels * 8, f_channels * 4, f, padding, activ_fn=activ_fn)
        self.decoder3 = DecoderBlock(f_channels * 4, f_channels * 2, f, padding, activ_fn=activ_fn)
        self.decoder4 = DecoderBlock(f_channels * 2, f_channels, f, padding, activ_fn=activ_fn)
        self.decoder_out = nn.Conv2d(f_channels, n_classes, kernel_size=1, padding=0)


    def forward(self, x):
        x = x.float()
        if next(self.parameters()).is_cuda:
            x = x.to('cuda')

        # encoder
        e1_pool, e1 = self.encoder1(x)
        e2_pool, e2 = self.encoder2(e1_pool)
        e3_pool, e3 = self.encoder3(e2_pool)
        e4_pool, e4 = self.encoder4(e3_pool)
        enc_out = self.encoder_out(e4_pool) #1024x28x28

        # decoder
        d1 = self.decoder1(enc_out, e4)
        d2 = self.decoder2(d1, e3)
        d3 = self.decoder3(d2, e2)
        d4 = self.decoder4(d3, e1)

        dec_out = self.decoder_out(d4)

        return dec_out