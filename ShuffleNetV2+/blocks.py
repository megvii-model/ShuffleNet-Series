import torch
import torch.nn as nn


class SELayer(nn.Module):

	def __init__(self, inplanes, isTensor=True):
		super(SELayer, self).__init__()
		if isTensor:
			# if the input is (N, C, H, W)
			self.SE_opr = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(inplanes // 4),
				nn.ReLU(inplace=True),
				nn.Conv2d(inplanes // 4, inplanes, kernel_size=1, stride=1, bias=False),
			)
		else:
			# if the input is (N, C)
			self.SE_opr = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Linear(inplanes, inplanes // 4, bias=False),
				nn.BatchNorm1d(inplanes // 4),
				nn.ReLU(inplace=True),
				nn.Linear(inplanes // 4, inplanes, bias=False),
			)

	def forward(self, x):
		atten = self.SE_opr(x)
		atten = torch.clamp(atten + 3, 0, 6) / 6
		return x * atten


class HS(nn.Module):

	def __init__(self):
		super(HS, self).__init__()

	def forward(self, inputs):
		clip = torch.clamp(inputs + 3, 0, 6) / 6
		return inputs * clip



class Shufflenet(nn.Module):

    def __init__(self, inp, oup, base_mid_channels, *, ksize, stride, activation, useSE):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert base_mid_channels == oup//2

        self.base_mid_channel = base_mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels, base_mid_channels, ksize, stride, pad, groups=base_mid_channels, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw-linear
            nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            None,
        ]
        if activation == 'ReLU':
            assert useSE == False
            '''This model should not have SE with ReLU'''
            branch_main[2] = nn.ReLU(inplace=True)
            branch_main[-1] = nn.ReLU(inplace=True)
        else:
            branch_main[2] = HS()
            branch_main[-1] = HS()
            if useSE:
                branch_main.append(SELayer(outputs))
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HS()
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

class Shuffle_Xception(nn.Module):

    def __init__(self, inp, oup, base_mid_channels, *, stride, activation, useSE):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]
        assert base_mid_channels == oup//2

        self.base_mid_channel = base_mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inp = inp
        outputs = oup - inp

        branch_main = [
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw
            nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels, base_mid_channels, 3, stride, 1, groups=base_mid_channels, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw
            nn.Conv2d(base_mid_channels, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels, base_mid_channels, 3, stride, 1, groups=base_mid_channels, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw
            nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            None,
        ]

        if activation == 'ReLU':
            branch_main[4] = nn.ReLU(inplace=True)
            branch_main[9] = nn.ReLU(inplace=True)
            branch_main[14] = nn.ReLU(inplace=True)
        else:
            branch_main[4] = HS()
            branch_main[9] = HS()
            branch_main[14] = HS()
        assert None not in branch_main

        if useSE:
            assert activation != 'ReLU'
            branch_main.append(SELayer(outputs))

        self.branch_main = nn.Sequential(*branch_main)

        if self.stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HS()
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]
