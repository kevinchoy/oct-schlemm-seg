"""Code for "K. C. Choy, G. Li, W. D. Stamer, S. Farsiu, Open-source deep learning-based automatic segmentation of
mouse Schlemm’s canal in optical coherence tomography images. Experimental Eye Research, 108844 (2021)."
Link: https://www.sciencedirect.com/science/article/pii/S0014483521004103
DOI: 10.1016/j.exer.2021.108844
The data and software here are only for research purposes. For licensing, please contact Duke University's Office of
Licensing & Ventures (OLV). Please cite our corresponding paper if you use this material in any form. You may not
redistribute our material without our written permission. """

import torch.nn as nn
import torch
# from models.archive.utils import unetConv2, UnetDsv2  # , UnetUp  # , UnetGridGatingSignal2
import torch.nn.functional as F
# from models.networks_other import init_weights
from .parts import DoubleConvBlock, UNetDsv2d, UNetUp, Conv2dBatchNormReLU, ResidualDilatedConvBlock, ResUNetUp


# from models.layers.grid_attention_layer import GridAttentionBlock2D


class LFDRAGUNet(nn.Module):
    # UNet2dFusedAttentionDsvMultiscaleResDilated
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=False, in_channels=1,
                 nonlocal_mode='concatenation', attention_dsample=(2, 2), is_batchnorm=True, encoder_scale=2):
        super(LFDRAGUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        encoder_filters = [filter//encoder_scale for filter in filters]

        # image pyramid
        self.scale_input = nn.AvgPool2d(kernel_size=(2, 2))

        self.inputs_scale2 = Conv2dBatchNormReLU(in_channels=self.in_channels, out_channels=encoder_filters[1],
                                                   kernel_size=(3, 3), padding=(1, 1))
        self.inputs_scale3 = Conv2dBatchNormReLU(in_channels=self.in_channels, out_channels=encoder_filters[2],
                                                   kernel_size=(3, 3), padding=(1, 1))
        self.inputs_scale4 = Conv2dBatchNormReLU(in_channels=self.in_channels, out_channels=encoder_filters[3],
                                                   kernel_size=(3, 3), padding=(1, 1))

        # self.inputs_2_scale2 = Conv2dBatchNormReLU(in_channels=self.in_channels, out_channels=filters[0] * 2,
        #                                            kernel_size=(3, 3), padding=(1, 1))
        # self.inputs_2_scale3 = Conv2dBatchNormReLU(in_channels=self.in_channels, out_channels=filters[1] * 2,
        #                                            kernel_size=(3, 3), padding=(1, 1))
        # self.inputs_2_scale4 = Conv2dBatchNormReLU(in_channels=self.in_channels, out_channels=filters[2] * 2,
        #                                            kernel_size=(3, 3), padding=(1, 1))


        self.inputs_2_scale2 = Conv2dBatchNormReLU(in_channels=self.in_channels, out_channels=encoder_filters[1],
                                                 kernel_size=(3, 3), padding=(1, 1))
        self.inputs_2_scale3 = Conv2dBatchNormReLU(in_channels=self.in_channels, out_channels=encoder_filters[2],
                                                 kernel_size=(3, 3), padding=(1, 1))
        self.inputs_2_scale4 = Conv2dBatchNormReLU(in_channels=self.in_channels, out_channels=encoder_filters[3],
                                                 kernel_size=(3, 3), padding=(1, 1))

        # downsampling
        self.conv1 = ResidualDilatedConvBlock(self.in_channels, encoder_filters[0], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = ResidualDilatedConvBlock(encoder_filters[0] + encoder_filters[1], filters[1] // encoder_scale, self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = ResidualDilatedConvBlock(encoder_filters[1] + encoder_filters[2], encoder_filters[2], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = ResidualDilatedConvBlock(encoder_filters[2] + encoder_filters[3], encoder_filters[3], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = ResidualDilatedConvBlock(encoder_filters[3], encoder_filters[4], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))

        # downsampling (2nd branch)
        self.conv1_2 = ResidualDilatedConvBlock(self.in_channels, encoder_filters[0], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool1_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2_2 = ResidualDilatedConvBlock(encoder_filters[0] + encoder_filters[1], filters[1] // encoder_scale, self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool2_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3_2 = ResidualDilatedConvBlock(encoder_filters[1] + encoder_filters[2], encoder_filters[2], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool3_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4_2 = ResidualDilatedConvBlock(encoder_filters[2] + encoder_filters[3], encoder_filters[3], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool4_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center_2 = ResidualDilatedConvBlock(encoder_filters[3], encoder_filters[4], self.is_batchnorm, kernel_size=(3, 3), padding=(1, 1))

        # ====

        # gating signal
        self.gating4 = UnetGridGatingSignal2(encoder_filters[4] * 2, filters[4], kernel_size=(1, 1),
                                             is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.fusedskip1 = ResidualDilatedConvBlock(in_channels=filters[0], out_channels=filters[0],
                                              kernel_size=(3, 3), stride=1, padding=1)
        self.fusedattention2 = FusedAttentionBlock(in_channels=encoder_filters[1], gate_channels=filters[2],
                                                   inter_channels=encoder_filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample, scale=encoder_scale)
        self.fusedattention3 = FusedAttentionBlock(in_channels=encoder_filters[2], gate_channels=filters[3],
                                                   inter_channels=encoder_filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample, scale=encoder_scale)
        self.fusedattention4 = FusedAttentionBlock(in_channels=encoder_filters[3], gate_channels=filters[4],
                                                   inter_channels=encoder_filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample, scale=encoder_scale)

        # ====

        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = UNetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = UNetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = UNetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # self.up_concat4 = ResUNetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        # self.up_concat3 = ResUNetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        # self.up_concat2 = ResUNetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        # self.up_concat1 = ResUNetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # deep supervision
        self.dsv4 = UNetDsv2d(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UNetDsv2d(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UNetDsv2d(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv2d(n_classes * 4, n_classes, 1)

    def forward(self, inputs):
        inputs_1 = inputs[:, 0].unsqueeze(1)  # Average
        inputs_2 = inputs[:, 1].unsqueeze(1)  # Speckle Variance

        inputs_scale_2 = self.scale_input(inputs_1)  # 256 x 256
        inputs_scale_3 = self.scale_input(inputs_scale_2)  # 128 x 128
        inputs_scale_4 = self.scale_input(inputs_scale_3)  # 64 x 64

        inputs_scale_2 = self.inputs_scale2(inputs_scale_2)  # bs x 32 x 256 x 256
        inputs_scale_3 = self.inputs_scale3(inputs_scale_3)  # bs x 64 x 128 x 128
        inputs_scale_4 = self.inputs_scale4(inputs_scale_4)  # bs x 128 x 64 x 64

        inputs_2_scale_2 = self.scale_input(inputs_2)  # 256 x 256
        inputs_2_scale_3 = self.scale_input(inputs_2_scale_2)  # 128 x 128
        inputs_2_scale_4 = self.scale_input(inputs_2_scale_3)  # 64 x 64

        inputs_2_scale_2 = self.inputs_2_scale2(inputs_2_scale_2)  # bs x 32 x 256 x 256
        inputs_2_scale_3 = self.inputs_2_scale3(inputs_2_scale_3)  # bs x 64 x 128 x 128
        inputs_2_scale_4 = self.inputs_2_scale4(inputs_2_scale_4)  # bs x 128 x 64 x 64

        # Feature Extraction 1
        conv1 = self.conv1(inputs_1)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(torch.cat((inputs_scale_2, maxpool1), dim=1))
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(torch.cat((inputs_scale_3, maxpool2), dim=1))
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(torch.cat((inputs_scale_4, maxpool3), dim=1))
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        # Feature Extraction 2
        conv1_2 = self.conv1_2(inputs_2)
        maxpool1_2 = self.maxpool1_2(conv1_2)

        conv2_2 = self.conv2_2(torch.cat((inputs_2_scale_2, maxpool1_2), dim=1))
        # conv2_2 = self.conv2_2(torch.cat((inputs_scale_2, maxpool1_2), dim=1))
        maxpool2_2 = self.maxpool2_2(conv2_2)

        conv3_2 = self.conv3_2(torch.cat((inputs_2_scale_3, maxpool2_2), dim=1))
        # conv3_2 = self.conv3_2(torch.cat((inputs_scale_3, maxpool2_2), dim=1))
        maxpool3_2 = self.maxpool3_2(conv3_2)

        conv4_2 = self.conv4_2(torch.cat((inputs_2_scale_4, maxpool3_2), dim=1))
        # conv4_2 = self.conv4_2(torch.cat((inputs_scale_4, maxpool3_2), dim=1))
        maxpool4_2 = self.maxpool4_2(conv4_2)

        center_2 = self.center_2(maxpool4_2)

        # Generate gating signal
        # gating = self.gating4(center)
        # gating_2 = self.gating4(center_2)
        gating = self.gating4(torch.cat([center, center_2], dim=1))

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att_fused4, att4, att4_2 = self.fusedattention4(conv4, conv4_2, gating)
        up4 = self.up_concat4(g_conv4, gating)

        g_conv3, att_fused3, att3, att3_2 = self.fusedattention3(conv3, conv3_2, up4)
        up3 = self.up_concat3(g_conv3, up4)

        g_conv2, att_fused2, att2, att2_2 = self.fusedattention2(conv2, conv2_2, up3)
        up2 = self.up_concat2(g_conv2, up3)

        skip1 = self.fusedskip1(torch.cat((conv1, conv1_2), dim=1))
        up1 = self.up_concat1(skip1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


# --------------------------------------------------------------------------------------


class FusedAttentionBlock(nn.Module):
    def __init__(self, in_channels, gate_channels, inter_channels, nonlocal_mode, sub_sample_factor, scale=1):
        super(FusedAttentionBlock, self).__init__()

        # Number of channels (pixel dimensions)
        self.gate_block_1 = GridAttentionBlock2D(in_channels=in_channels, gating_channels=gate_channels,
                                                 inter_channels=inter_channels, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        # self.combine_gates_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        #                                    nn.BatchNorm2d(in_channels),
        #                                    nn.ReLU(inplace=True)
        #                                    )
        self.gate_block_2 = GridAttentionBlock2D(in_channels=in_channels, gating_channels=gate_channels,
                                                 inter_channels=inter_channels, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)

        self.combine_gates = nn.Sequential(nn.Conv2d(in_channels * scale, in_channels * scale, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(in_channels * scale),
                                           nn.ReLU(inplace=True)
                                           )
        # Output transform
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels*scale, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels*scale),
        )

        # self.weight_attention = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1), stride=1, padding=0,
        #                                     bias=False)

        # self.log_weight = nn.Parameter(torch.log(0.5))
        self.log_weight1 = nn.Parameter(torch.log(torch.tensor(0.5)))
        self.log_weight2 = nn.Parameter(torch.log(torch.tensor(0.5)))

    def forward(self, x1, x2, gating_signal):
        sigm_psi_f_1 = self.gate_block_1(x1, gating_signal)
        sigm_psi_f_2 = self.gate_block_2(x2, gating_signal)

        # weighted average
        weighted_sigm_psi_f_1 = sigm_psi_f_1 * self.log_weight1.exp()
        weighted_sigm_psi_f_2 = sigm_psi_f_2 * self.log_weight2.exp()
        sigm_psi_f = (weighted_sigm_psi_f_1 + weighted_sigm_psi_f_2) / (self.log_weight1.exp() + self.log_weight2.exp())

        x = torch.cat([x1, x2], dim=1)
        y = sigm_psi_f.expand_as(x) * x
        g_input = self.W(y)

        gate = self.combine_gates(g_input)

        return gate, sigm_psi_f, sigm_psi_f_1, sigm_psi_f_2


# -------------------------------------------------------------------------------------------------


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
        attention_1 = self.gate_block_1(input, gating_signal)
        # gates = self.combine_gates(gate_1)
        return attention_1


class UnetGridGatingSignal2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1, 1), is_batchnorm=True):
        super(UnetGridGatingSignal2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1, 1), (0, 0)),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1, 1), (0, 0)),
                                       nn.ReLU(inplace=True),
                                       )

    def forward(self, inputs):
        # inputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv1(inputs)
        return outputs


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        # elif mode == 'concatenation_debug':
        #     self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        # y = sigm_psi_f.expand_as(x) * x
        # W_y = self.W(y)

        # return W_y, sigm_psi_f
        return sigm_psi_f


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )


if __name__ == '__main__':
    from torch.autograd import Variable

    mode_list = ['concatenation']

    for mode in mode_list:
        img = Variable(torch.rand(2, 16, 10, 10, 10))
        gat = Variable(torch.rand(2, 64, 4, 4, 4))
        net = GridAttentionBlock2D(in_channels=16, inter_channels=16, gating_channels=64, mode=mode,
                                   sub_sample_factor=(2, 2, 2))
        out, sigma = net(img, gat)
        print(out.size())
