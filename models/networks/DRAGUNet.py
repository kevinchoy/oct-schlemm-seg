import torch.nn as nn
import torch
from .parts import UNetGridGatingSignal2d, UNetDsv2d, UNetUp, GridAttentionBlock2D, \
    ResidualDilatedConvBlock


class DRAGUNet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=2,
                 nonlocal_mode='concatenation', attention_dsample=(2, 2), is_batchnorm=True):
        super(DRAGUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # image pyramid
        self.scale_input = nn.AvgPool2d(kernel_size=(2, 2))
        self.input_scale2 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=filters[0] * 2,
                                                    kernel_size=(3, 3), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(filters[0] * 2),
                                          nn.ReLU(inplace=True)
                                          )
        self.input_scale3 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=filters[1] * 2,
                                                    kernel_size=(3, 3), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(filters[1] * 2),
                                          nn.ReLU(inplace=True)
                                          )
        self.input_scale4 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=filters[2] * 2,
                                                    kernel_size=(3, 3), padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(filters[2] * 2),
                                          nn.ReLU(inplace=True)
                                          )
        # downsampling
        self.conv1 = ResidualDilatedConvBlock(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = ResidualDilatedConvBlock(filters[0] * 3, filters[1], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = ResidualDilatedConvBlock(filters[1] * 3, filters[2], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv4 = ResidualDilatedConvBlock(filters[2] * 3, filters[3], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.center = ResidualDilatedConvBlock(filters[3], filters[4], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.gating = UNetGridGatingSignal2d(filters[4], filters[4], kernel_size=(1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = UNetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = UNetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = UNetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)
        # deep supervision
        self.dsv4 = UNetDsv2d(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UNetDsv2d(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UNetDsv2d(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv2d(n_classes * 4, n_classes, 1)

    def forward(self, inputs):
        # Feature Extraction
        inputs_scale_2 = self.scale_input(inputs)  # 256 x 256
        inputs_scale_3 = self.scale_input(inputs_scale_2)  # 128 x 128
        inputs_scale_4 = self.scale_input(inputs_scale_3)  # 64 x 64

        inputs_scale_2 = self.input_scale2(inputs_scale_2)  # bs x 32 x 256 x 256
        inputs_scale_3 = self.input_scale3(inputs_scale_3)  # bs x 64 x 128 x 128
        inputs_scale_4 = self.input_scale4(inputs_scale_4)  # bs x 128 x 64 x 64

        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(torch.cat((inputs_scale_2, maxpool1), dim=1))
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(torch.cat((inputs_scale_3, maxpool2), dim=1))
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(torch.cat((inputs_scale_4, maxpool3), dim=1))
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))

        return final


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1
