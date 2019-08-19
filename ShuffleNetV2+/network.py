import torch
import torch.nn as nn
from blocks import Shufflenet, Shuffle_Xception, HS, SELayer

class ShuffleNetV2_Plus(nn.Module):
    def __init__(self, input_size=224, n_class=1000, architecture=None, model_size='Large'):
        super(ShuffleNetV2_Plus, self).__init__()

        print('model size is ', model_size)

        assert input_size % 32 == 0
        assert architecture is not None

        self.stage_repeats = [4, 4, 8, 4]
        if model_size == 'Large':
            self.stage_out_channels = [-1, 16, 68, 168, 336, 672, 1280]
        elif model_size == 'Medium':
            self.stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]
        elif model_size == 'Small':
            self.stage_out_channels = [-1, 16, 36, 104, 208, 416, 1280]
        else:
            raise NotImplementedError


        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            HS(),
        )

        self.features = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            activation = 'HS' if idxstage >= 1 else 'ReLU'
            useSE = 'True' if idxstage >= 2 else False

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                blockIndex = architecture[archIndex]
                archIndex += 1
                if blockIndex == 0:
                    print('Shuffle3x3')
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=3, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 1:
                    print('Shuffle5x5')
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=5, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 2:
                    print('Shuffle7x7')
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=7, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 3:
                    print('Xception')
                    self.features.append(Shuffle_Xception(inp, outp, base_mid_channels=outp // 2, stride=stride,
                                    activation=activation, useSE=useSE))
                else:
                    raise NotImplementedError
                input_channel = output_channel
        assert archIndex == len(architecture)
        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            HS()
        )
        self.globalpool = nn.AvgPool2d(7)
        self.LastSE = SELayer(1280)
        self.fc = nn.Sequential(
            nn.Linear(1280, 1280, bias=False),
            HS(),
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(1280, n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        x = self.LastSE(x)

        x = x.contiguous().view(-1, 1280)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
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

if __name__ == "__main__":
    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    model = ShuffleNetV2_Plus(architecture=architecture)
    # print(model)

    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())
