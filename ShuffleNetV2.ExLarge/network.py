import torch
import torch.nn as nn


class Conv_BN_ReLU(nn.Module):

    def __init__(self, in_channel, out_channel, k_size, stride=1, padding=0, groups=1,
                 has_bn=True, has_relu=True, gaussian_init=False):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=k_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=False)
        if gaussian_init:
            nn.init.normal_(self.conv.weight.data, 0, 0.01)

        if has_bn:
            self.bn = nn.BatchNorm2d(out_channel)

        self.has_bn = has_bn
        self.has_relu = has_relu
        if has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        nn.init.normal_(self.fc.weight.data, 0, 0.01)

    def forward(self, x):
        return self.fc(x)


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


class ShuffleV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, stride, groups, has_proj=False, has_se=False):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.has_proj = has_proj
        self.has_se = has_se
        self.relu = nn.ReLU(inplace=True)

        if has_proj:
            self.proj = Conv_BN_ReLU(in_channels, out_channels - mid_channels, k_size=3, stride=stride, padding=1,
                                     has_bn=True, has_relu=True)

        self.branch_main = nn.Sequential(
            Conv_BN_ReLU(in_channels, out_channels, k_size=1, stride=1, padding=0,
                         has_bn=True, has_relu=True),
            Conv_BN_ReLU(out_channels, out_channels, k_size=3, stride=stride, padding=1, groups=groups,
                         has_bn=True, has_relu=True),
            Conv_BN_ReLU(out_channels, out_channels, k_size=3, stride=1, padding=1, groups=out_channels,
                         has_bn=True, has_relu=False),
            Conv_BN_ReLU(out_channels, mid_channels, k_size=1, stride=1, padding=0,
                         has_bn=True, has_relu=False),
        )

        if has_se:
            self.se_globalpool = nn.AdaptiveAvgPool2d(output_size=1)
            self.se_fc1 = FC(mid_channels, mid_channels // 4)
            self.se_fc2 = FC(mid_channels // 4, mid_channels)
            se_block = [
                self.se_fc1,
                nn.ReLU(inplace=True),
                self.se_fc2,
                nn.Sigmoid(),
            ]
            self.se_block = nn.Sequential(*se_block)

    def forward(self, old_x):
        if self.has_proj:
            proj, x = old_x, old_x
        else:
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
            x = self.relu(x_proj + x)

        x = torch.cat((proj, x), dim=1)

        return x

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, model_size='ExLarge'):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == 'ExLarge':
            self.pre = [2, 3, 4, 5]
            self.stage_repeats = [8, 16, 36, 10]
            self.outputs = [320, 640, 1280, 2560]
            self.enable_stride = [False, True, True, True]
        else:
            raise NotImplementedError

        self.first_conv = nn.Sequential(
            Conv_BN_ReLU(3, 64, k_size=3, stride=2, padding=1),
            Conv_BN_ReLU(64, 128, k_size=3, stride=1, padding=1),
            Conv_BN_ReLU(128, 256, k_size=3, stride=1, padding=1),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = nn.ModuleList()
        input_channel = 256
        if model_size == 'ExLarge':
            for p, s, o, es in zip(self.pre, self.stage_repeats, self.outputs, self.enable_stride):
                feature = []
                for i in range(s):
                    prefix = "{}{}".format(p, str(i))
                    stride = 1 if not es or i > 0 else 2
                    has_proj = False if i > 0 else True
                    feature.append(ShuffleV2Block(in_channels=input_channel, out_channels=o, mid_channels=o // 2,
                                                      stride=stride, groups=16, has_proj=has_proj, has_se=True))
                    input_channel = o // 2
                feature.append(Conv_BN_ReLU(o, o, k_size=1, stride=1, padding=0))
                input_channel = o
                feature = nn.Sequential(*feature)
                self.features.append(feature)
                if p == 2:
                    self.predict_56 = ExtraLabelPredict(in_channels=320, out_channels=256)
                elif p == 3:
                    self.predict_28 = ExtraLabelPredict(in_channels=640, out_channels=512)
                elif p == 4:
                    self.predict_14 = ExtraLabelPredict(in_channels=1280, out_channels=1024)

        self.globalpool = nn.AvgPool2d(7)
        if self.model_size == 'ExLarge':
            self.dropout = nn.Dropout(0.2)
        self.fc = FC(2560, n_class)

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
        # 1 * 256 * 56 * 56

        x = self.features[0](x)
        # 1 * 320 * 56 * 56
        if self.training:
            predict_56 = self.predict_56(x)

        x = self.features[1](x)
        # 1 * 640 * 28 * 28
        if self.training:
            predict_28 = self.predict_28(x)

        x = self.features[2](x)
        # 1 * 1280 * 14 * 14
        if self.training:
            predict_14 = self.predict_14(x)

        x = self.features[3](x)
        # 1 * 2560 * 7 * 7

        x = self.globalpool(x)
        if self.model_size == 'ExLarge':
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
    return model


if __name__ == "__main__":
    create_network()
