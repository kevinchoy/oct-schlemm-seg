"""Code for "K. C. Choy, G. Li, W. D. Stamer, S. Farsiu, Open-source deep learning-based automatic segmentation of
mouse Schlemm’s canal in optical coherence tomography images. Experimental Eye Research, 108844 (2021)."
Link: https://www.sciencedirect.com/science/article/pii/S0014483521004103
DOI: 10.1016/j.exer.2021.108844
The data and software here are only for research purposes. For licensing, please contact Duke University's Office of
Licensing & Ventures (OLV). Please cite our corresponding paper if you use this material in any form. You may not
redistribute our material without our written permission. """

import torch
import torch.nn as nn
import torch.nn.functional as F


# from models.networks_other import init_weights


# U-Net parts
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm, n=2, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.n = n
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, s, p, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_channels = out_channels

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_channels = out_channels

        # # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UNetUp, self).__init__()
        # self.conv = unetConv2(in_size, out_size, False)
        self.conv = DoubleConvBlock(in_size, out_size, is_batchnorm)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1),
                                    )
            # self.conv = DoubleConvBlock(in_size + out_size, out_size, is_batchnorm)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UNetDsv2d(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UNetDsv2d, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
                                 )

    def forward(self, input):
        return self.dsv(input)


# building blocks
class Conv2dBatchNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(Conv2dBatchNormReLU, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                                      nn.BatchNorm2d(int(out_channels)),
                                      nn.ReLU(inplace=True),
                                      )

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Conv2dBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=False):
        super(Conv2dBatchNorm, self).__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                     nn.BatchNorm2d(int(n_filters)), )

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class Deconv2dBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=False):
        super(Deconv2dBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                         padding=padding, stride=stride, bias=bias),
                                      nn.BatchNorm2d(int(n_filters)), )

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class Deconv2dBatchNormReLU(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=False):
        super(Deconv2dBatchNormReLU, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                          padding=padding, stride=stride, bias=bias),
                                       nn.BatchNorm2d(int(n_filters)),
                                       nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


# ==== Gating ====

class UNetGatingSignal2d(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UNetGatingSignal2d, self).__init__()
        self.fmap_size = (4, 4)

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, in_size // 2, (1, 1), (1, 1), (0, 0), bias=False),
                                       nn.BatchNorm2d(in_size // 2),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool2d(output_size=self.fmap_size),
                                       )
            self.fc1 = nn.Linear(in_features=(in_size // 2) * self.fmap_size[0] * self.fmap_size[1],
                                 out_features=out_size, bias=True)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, in_size // 2, (1, 1), (1, 1), (0, 0)),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool2d(output_size=self.fmap_size),
                                       )
            self.fc1 = nn.Linear(in_features=(in_size // 2) * self.fmap_size[0] * self.fmap_size[1],
                                 out_features=out_size, bias=True)

        # # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        batch_size = inputs.size(0)
        outputs = self.conv1(inputs)
        outputs = outputs.view(batch_size, -1)
        outputs = self.fc1(outputs)
        return outputs


class UNetGridGatingSignal2d(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1, 1), is_batchnorm=True):
        super(UNetGridGatingSignal2d, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1, 1), (0, 0), bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1, 1), (0, 0)),
                                       nn.ReLU(inplace=True),
                                       )

        # # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = []
            # for j in i:
            #     for index in range(len(j)):
            #         self.inputs.append(j[index].data.clone())
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale:
            self.rescale_output_array(x.size())

        return self.inputs, self.outputs


# ==== Attention Block ====

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

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0,
                    bias=False),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # # Initialise weights
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
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
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )


# ==== Residual U-Net parts ====

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm=True, n=2, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_size = in_channels
        self.out_size = out_channels
        self.n = n
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        assert is_batchnorm is True
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.BatchNorm2d(out_channels),  # pre-activation
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels, out_channels, kernel_size, s, p), )

                # conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p, bias=False),  # post-activation
                #                      nn.BatchNorm2d(out_size),
                #                      nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                # in_size = out_size

        # else:
        #     for i in range(1, n + 1):
        #         conv = nn.Sequential(nn.ReLU(inplace=True),
        #                              nn.Conv2d(in_size, out_size, ks, s, p), )
        #         setattr(self, 'conv%d' % i, conv)
        #         in_size = out_size
        self.conv1x1 = nn.Conv2d(in_channels=self.in_size, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, inputs):
        identity = self.conv1x1(inputs)

        x = identity

        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x + identity


class ResUNetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super().__init__()
        inter_size = in_size // 2
        self.combine = nn.Sequential(
            nn.Conv2d(in_size, inter_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=inter_size),
            nn.ReLU(inplace=True)
        )
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
            self.conv = ResidualConvBlock(inter_size, out_size, is_batchnorm)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv = ResidualConvBlock(inter_size, out_size, is_batchnorm)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        combined = self.combine(torch.cat([outputs1, outputs2], 1))
        return self.conv(combined)


# ====  Residual Dilated Conv U-Net parts ====
class ResidualDilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm=True, n=2, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_size = in_channels
        self.out_size = out_channels
        self.n = n
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        assert is_batchnorm is True

        for i in range(1, n + 1):
            conv = nn.Sequential(nn.BatchNorm2d(out_channels),  # pre-activation
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels, out_channels, kernel_size, s, p), )
            setattr(self, 'conv%d_d1' % i, conv)

        for i in range(1, n + 1):
            conv = nn.Sequential(nn.BatchNorm2d(out_channels),  # pre-activation
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels, out_channels, kernel_size, s, padding=3, dilation=3), )
            setattr(self, 'conv%d_d3' % i, conv)

        for i in range(1, n + 1):
            conv = nn.Sequential(nn.BatchNorm2d(out_channels),  # pre-activation
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels, out_channels, kernel_size, s, padding=5, dilation=5), )
            setattr(self, 'conv%d_d5' % i, conv)

        self.conv1x1 = nn.Conv2d(in_channels=self.in_size, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, inputs):
        x = self.conv1x1(inputs)

        for i in range(1, self.n + 1):
            conv_d1 = getattr(self, 'conv%d_d1' % i)
            conv_d3 = getattr(self, 'conv%d_d3' % i)
            conv_d5 = getattr(self, 'conv%d_d5' % i)

            if i == 1:
                x1 = conv_d1(x)
                x3 = conv_d3(x)
                x5 = conv_d5(x)
            else:
                x1 = conv_d1(x1)
                x3 = conv_d3(x3)
                x5 = conv_d5(x5)

        return x + x1 + x3 + x5


# ==== Squeeze-Excite block ====
class SEBlock(nn.Module):
    def __init__(self, n_channel, ratio):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(n_channel, n_channel // ratio, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channel, n_channel // ratio, kernel_size=1)
        self.sigmoid = nn.sigmoid()

    def forward(self, x):
        out = self.squeeze(x)
        out = self.relu(self.conv1(out))
        out = self.sigmoid(self.conv2(out))
        return out * x
