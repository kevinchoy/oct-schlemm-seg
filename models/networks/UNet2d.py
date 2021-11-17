# adapted from https://github.com/ozan-oktay/Attention-Gated-Networks
import torch.nn as nn
from .parts import DoubleConvBlock, UNetUp

class UNet2d(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=2, is_batchnorm=True):
        super(UNet2d, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = DoubleConvBlock(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = DoubleConvBlock(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = DoubleConvBlock(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = DoubleConvBlock(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = DoubleConvBlock(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = UNetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = UNetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = UNetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16 x 512 x 512
        maxpool1 = self.maxpool1(conv1)  # 16 x 256 x 256

        conv2 = self.conv2(maxpool1)  # 32 x 256 x 256
        maxpool2 = self.maxpool2(conv2)  # 32 x 128 x 128

        conv3 = self.conv3(maxpool2)  # 64 x 128 x 128
        maxpool3 = self.maxpool3(conv3)  # 64 x 64 x 64

        conv4 = self.conv4(maxpool3)  # 128 x 64 x 64
        maxpool4 = self.maxpool4(conv4)  # 128 x 32 x 32

        center = self.center(maxpool4)  # 256 x 32 x 32

        up4 = self.up_concat4(conv4, center)  # 128 x 64 x 64
        up3 = self.up_concat3(conv3, up4)  # 64 x 128 x 128
        up2 = self.up_concat2(conv2, up3)  # 32 x 256 x 256
        up1 = self.up_concat1(conv1, up2)  # 16 x 512 x 512

        final = self.final(up1)  # 1 x 512 x 512

        return final

