# -*- coding: utf-8 -*-
# @Time    : 2019-08-02 18:48 
# @Author  : Yi Zou
# @File    : network.py 
# @Software: PyCharm


import torch
import torch.nn as nn
from flops_counter import print_profile


class Conv_BN_ReLU(nn.Module):

    def __init__(self, in_channel, out_channel, k_size, stride=1, padding=0, groups=1, has_bn=True, has_relu=True):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=k_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-9)
        self.has_bn = has_bn

        self.has_relu = has_relu
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class ShuffleV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, has_proj=False, has_se=False):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.has_proj = has_proj
        self.has_se = has_se
        self.relu = nn.ReLU(inplace=True)

        if stride == 2:
            self.down = Conv_BN_ReLU(out_channels * 2, out_channels * 2, k_size=1, stride=1, padding=0)

        if has_proj:
            self.proj = Conv_BN_ReLU(in_channels, out_channels, k_size=3, stride=stride, padding=1, has_bn=True, has_relu=False)

        self.branch_main = nn.Sequential(
            Conv_BN_ReLU(in_channels, out_channels, k_size=1, stride=1, padding=0, has_bn=True, has_relu=True),
            Conv_BN_ReLU(out_channels, out_channels, k_size=3, stride=stride, padding=1, groups=groups, has_bn=True, has_relu=True),
            Conv_BN_ReLU(out_channels, out_channels, k_size=1, stride=1, padding=0, has_bn=True, has_relu=False),
        )

        if has_se:
            self.se_globalpool = nn.AdaptiveAvgPool2d(output_size=1)
            self.se_fc1 = nn.Linear(out_channels, out_channels)
            self.se_fc2 = nn.Linear(out_channels, out_channels)
            se_block = [
                self.se_fc1,
                nn.ReLU(inplace=True),
                self.se_fc2,
                nn.Sigmoid(),
            ]
            self.se_block = nn.Sequential(*se_block)

    def forward(self, old_x):
        proj, x = self.channel_shuffle(old_x)
        x_proj = x
        if self.has_proj:
            proj = self.proj(proj)

        x = self.branch_main(x)

        if self.has_se:
            se_scale = self.se_globalpool(x).view(x.size(0), -1)
            se_scale = self.se_block(se_scale).unsqueeze(-1).unsqueeze(-1)
            x = x * se_scale

        if not self.has_proj:
            x = x_proj + x

        x = self.relu(torch.cat((proj, x), dim=1))

        if self.stride == 2:
            x = self.down(x)

        return x

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ExtraLabelPredict(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=1000):
        super(ExtraLabelPredict, self).__init__()
        self.num_classes = num_classes
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Sequential(
            Conv_BN_ReLU(in_channels, out_channels,  1, 1, 0),
            Conv_BN_ReLU(out_channels, out_channels, 3, 1, 1)
        )
        self.globalpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, inputs):
        inputs = self.maxpool(inputs)
        inputs = self.conv(inputs)
        inputs = self.globalpool(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        inputs = self.fc(inputs)
        return inputs


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, model_size='large'):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == 'large':
            self.pre = [2, 3, 4, 5]
            self.stage_repeats = [10, 10, 23, 10]
            self.mid_outputs = [64, 128, 256, 512]
            self.enable_stride = [False, True, True, True]
        else:
            raise NotImplementedError

        self.first_conv = nn.Sequential(
            Conv_BN_ReLU(3, 64, k_size=3, stride=2, padding=1),
            Conv_BN_ReLU(64, 64, k_size=3, stride=1, padding=1),
            Conv_BN_ReLU(64, 128, k_size=3, stride=1, padding=1),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = nn.ModuleList()
        input_channel = 64
        if model_size == 'large':
            for p, s, o, es in zip(self.pre, self.stage_repeats, self.mid_outputs, self.enable_stride):
                feature = nn.Sequential()
                for i in range(s):
                    prefix = "{}{}".format(p, chr(ord("a") + i))
                    stride = 1 if not es or i > 0 else 2
                    has_proj = False if i > 0 else True
                    feature.add_module(prefix, ShuffleV2Block(input_channel, o * 2, stride, groups=8, has_proj=has_proj, has_se=True))
                    input_channel = o * 2
                self.features.append(feature)
                if p == 2:
                    self.predict_56 = ExtraLabelPredict(in_channels=256, out_channels=256)
                elif p == 3:
                    self.predict_28 = ExtraLabelPredict(in_channels=512, out_channels=512)
                elif p == 4:
                    self.predict_14 = ExtraLabelPredict(in_channels=1024, out_channels=1024)

        self.conv_last = Conv_BN_ReLU(input_channel * 2, 1280, 3, 1, 1)
        self.globalpool = nn.AvgPool2d(7)
        if self.model_size == 'large':
            self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, n_class)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        # 5 * 128 * 56 * 56

        x = self.features[0](x)
        # 5 * 256 * 56 * 56
        if self.training:
            predict_56 = self.predict_56(x)

        x = self.features[1](x)
        # 5 * 512 * 28 * 28
        if self.training:
            predict_28 = self.predict_28(x)

        x = self.features[2](x)
        # 5 * 1024 * 14 * 14
        if self.training:
            predict_14 = self.predict_14(x)

        x = self.features[3](x)
        # 5 * 2048 * 7 * 7

        x = self.conv_last(x)
        x = self.globalpool(x)
        if self.model_size == 'large':
            x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        if self.training:
            # Loss is scaled by 1.0, 0.7, 0.5, 0.3
            return x, predict_14, predict_28, predict_56
        else:
            return x


def create_network():
    model = ShuffleNetV2()
    print_profile(model)
    return model


if __name__ == "__main__":
    create_network()




