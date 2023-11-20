import torch
import torch.nn as nn
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class multitask_network(nn.Module):
    def __init__(self, n_channels, n_seg, n_landmark):
        super(multitask_network, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder
        self.Conv1 = conv_block(ch_in=n_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # decoder seg
        self.Up5_1 = up_conv(ch_in=1024, ch_out=512)
        self.Att5_1 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5_1 = conv_block(ch_in=1024, ch_out=512)

        self.Up4_1 = up_conv(ch_in=512, ch_out=256)
        self.Att4_1 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4_1 = conv_block(ch_in=768, ch_out=256)

        self.Up3_1 = up_conv(ch_in=256, ch_out=128)
        self.Att3_1 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3_1 = conv_block(ch_in=384, ch_out=128)

        self.Up2_1 = up_conv(ch_in=128, ch_out=64)
        self.Att2_1 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2_1 = conv_block(ch_in=192, ch_out=64)

        # decoder lm
        self.Up5_2 = up_conv(ch_in=1024, ch_out=512)
        self.Att5_2 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5_2 = conv_block(ch_in=1024, ch_out=512)

        self.Up4_2 = up_conv(ch_in=512, ch_out=256)
        self.Att4_2 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4_2 = conv_block(ch_in=768, ch_out=256)

        self.Up3_2 = up_conv(ch_in=256, ch_out=128)
        self.Att3_2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3_2 = conv_block(ch_in=384, ch_out=128)

        self.Up2_2 = up_conv(ch_in=128, ch_out=64)
        self.Att2_2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2_2 = conv_block(ch_in=192, ch_out=64)

        # exchange
        self.stage4_seg_to_lm = conv_block(ch_in=256, ch_out=256)
        self.stage4_lm_to_seg = conv_block(ch_in=256, ch_out=256)

        self.stage3_seg_to_lm = conv_block(ch_in=128, ch_out=128)
        self.stage3_lm_to_seg = conv_block(ch_in=128, ch_out=128)

        self.stage2_seg_to_lm = conv_block(ch_in=64, ch_out=64)
        self.stage2_lm_to_seg = conv_block(ch_in=64, ch_out=64)

        # output
        self.Conv_1x1_sg = nn.Conv2d(64, n_seg, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_lm = nn.Conv2d(64, n_landmark, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x5_lm = x5
        # decoding + concat path
        # stage5
        d5_1 = self.Up5_1(x5)
        x4_1 = self.Att5_1(g=d5_1, x=x4)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d5_2 = self.Up5_2(x5_lm)
        x4_2 = self.Att5_2(g=d5_2, x=x4)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)

        # stage4
        d4_1 = self.Up4_1(d5_1)
        d4_2 = self.Up4_2(d5_2)

        stage4_seg_to_lm = self.stage4_seg_to_lm(d4_1)
        stage4_lm_to_seg = self.stage4_lm_to_seg(d4_2)

        x3_1 = self.Att4_1(g=d4_1, x=x3)
        x3_2 = self.Att4_2(g=d4_2, x=x3)

        d4_1 = torch.cat((x3_1, stage4_lm_to_seg, d4_1), dim=1)
        d4_2 = torch.cat((x3_2, stage4_seg_to_lm, d4_2), dim=1)

        d4_1 = self.Up_conv4_1(d4_1)
        d4_2 = self.Up_conv4_2(d4_2)

        # stage3
        d3_1 = self.Up3_1(d4_1)
        d3_2 = self.Up3_2(d4_2)

        stage3_seg_to_lm = self.stage3_seg_to_lm(d3_1)
        stage3_lm_to_seg = self.stage3_lm_to_seg(d3_2)

        x2_1 = self.Att3_1(g=d3_1, x=x2)
        x2_2 = self.Att3_2(g=d3_2, x=x2)

        d3_1 = torch.cat((x2_1, stage3_lm_to_seg, d3_1), dim=1)
        d3_2 = torch.cat((x2_2, stage3_seg_to_lm, d3_2), dim=1)

        d3_1 = self.Up_conv3_1(d3_1)
        d3_2 = self.Up_conv3_2(d3_2)

        # stage2
        d2_1 = self.Up2_1(d3_1)
        d2_2 = self.Up2_2(d3_2)

        stage2_seg_to_lm = self.stage2_seg_to_lm(d2_1)
        stage2_lm_to_seg = self.stage2_lm_to_seg(d2_2)

        x1_1 = self.Att2_1(g=d2_1, x=x1)
        x1_2 = self.Att2_2(g=d2_2, x=x1)

        d2_1 = torch.cat((x1_1, stage2_lm_to_seg, d2_1), dim=1)
        d2_2 = torch.cat((x1_2, stage2_seg_to_lm, d2_2), dim=1)

        d2_1 = self.Up_conv2_1(d2_1)
        d2_2 = self.Up_conv2_2(d2_2)

        # output
        seg = self.Conv_1x1_sg(d2_1)
        lm = self.Conv_1x1_lm(d2_2)

        return torch.sigmoid(lm), seg

