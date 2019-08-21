import torch.nn as nn
from blocks import ConvBNReLU, FC, ShuffleNetV2BlockSearched


class ShuffleNetV2DetNAS(nn.Module):
    def __init__(self, n_class=1000, model_size='300M'):
        super(ShuffleNetV2DetNAS, self).__init__()
        print('Model size is {}.'.format(model_size))

        if model_size == '3.8G':
            architecture = [0, 0, 3, 1, 2, 1, 0, 2, 0, 3, 1, 2, 3, 3, 2, 0, 2, 1, 1, 3,
                            2, 0, 2, 2, 2, 1, 3, 1, 0, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3]
            stage_repeats = [8, 8, 16, 8]
            stage_out_channels = [-1, 72, 172, 432, 864, 1728, 1728]
        elif model_size == '1.3G':
            architecture = [0, 0, 3, 1, 2, 1, 0, 2, 0, 3, 1, 2, 3, 3, 2, 0, 2, 1, 1, 3,
                            2, 0, 2, 2, 2, 1, 3, 1, 0, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3]
            stage_repeats = [8, 8, 16, 8]
            stage_out_channels = [-1, 48, 96, 240, 480, 960, 1024]
        elif model_size == '300M':
            architecture = [2, 1, 0, 3, 1, 3, 0, 3, 2, 0, 1, 1, 3, 3, 3, 3, 3, 3, 3, 1]
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        else:
            raise NotImplementedError

        self.first_conv = ConvBNReLU(in_channel=3, out_channel=stage_out_channels[1], k_size=3, stride=2, padding=1, gaussian_init=True)

        self.features = list()

        in_channels = stage_out_channels[1]
        i_th = 0
        for id_stage in range(1, len(stage_repeats) + 1):
            out_channels = stage_out_channels[id_stage + 1]
            repeats = stage_repeats[id_stage - 1]
            for id_repeat in range(repeats):
                prefix = str(id_stage) + chr(ord('a') + id_repeat)
                stride = 1 if id_repeat > 0 else 2
                self.features.append(ShuffleNetV2BlockSearched(prefix, in_channels=in_channels, out_channels=out_channels,
                                                               stride=stride, base_mid_channels=out_channels // 2, i_th=i_th,
                                                               architecture=architecture))
                in_channels = out_channels
                i_th += 1

        self.features = nn.Sequential(*self.features)

        self.last_conv = ConvBNReLU(in_channel=in_channels, out_channel=stage_out_channels[-1], k_size=1, stride=1, padding=0)
        self.drop_out = nn.Dropout2d(p=0.2)
        self.global_pool = nn.AvgPool2d(7)
        self.fc = FC(in_channels=stage_out_channels[-1], out_channels=n_class)
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

        x = self.features(x)

        x = self.last_conv(x)
        x = self.drop_out(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_network():
    model = ShuffleNetV2DetNAS(model_size='1.3G')
    print(model)
    return model


if __name__ == "__main__":
    create_network()

