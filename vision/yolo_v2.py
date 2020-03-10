import torch
import torch.nn as nn
import torch.nn.functional as F


class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(SeparableConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.depth_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                    groups=self.in_channels)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.point_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=(1, 1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.point_conv(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(BasicConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size,
                              stride=self.stride, padding=self.padding)
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class YoloV2(nn.Module):
    def __init__(self, class_num, anchor_num=3, target_size=(416, 416)):
        super(YoloV2, self).__init__()
        self.class_num = class_num
        self.anchor_num = anchor_num
        self.target_size = target_size
        self.out_channel = (self.class_num + 5) * self.anchor_num
        self.conv1 = nn.Sequential(BasicConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=3//2))
        self.conv2 = nn.Sequential(BasicConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3//2))
        self.conv3 = nn.Sequential(BasicConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3//2),
                                   BasicConvBlock(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=1//2),
                                   BasicConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3//2))
        self.conv4 = nn.Sequential(BasicConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3//2),
                                   BasicConvBlock(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=1//2),
                                   BasicConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3//2))
        self.conv5 = nn.Sequential(BasicConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                                   BasicConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                                   BasicConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
                                   BasicConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
                                   BasicConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2))
        self.conv6 = nn.Sequential(BasicConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                                   BasicConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
                                   BasicConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
                                   BasicConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
                                   BasicConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2))
        self.passthrough = SpaceToDepth(block_size=2)
        self.netout = nn.Sequential(BasicConvBlock(in_channels=1024 + 2048, out_channels=self.out_channel, kernel_size=3, stride=1, padding=3//2),
                                    nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=1//2, bias=False))

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv3(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv4(x), kernel_size=2, stride=2)
        conv_5 = self.conv5(x)
        x = self.conv6(F.max_pool2d(conv_5, kernel_size=2, stride=2))
        passthrough = self.passthrough(conv_5)
        x = torch.cat([passthrough, x], dim=1)
        netout = self.netout(x)
        netout = netout.permute(0, 2, 3, 1).contiguous()
        return netout
