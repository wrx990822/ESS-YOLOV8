import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  

    if d > 1:

        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:

        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):  

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):

    default_act = SiLU() 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class GhostBaseConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(GhostBaseConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Involution(nn.Module):

    def __init__(self, c1,  kernel_size, stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.c1 = c1
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.c1 // self.group_channels
        self.conv1 = nn.Conv2d(
            c1, c1 // reduction_ratio, 1)
        self.conv2 = nn.Conv2d(
            c1 // reduction_ratio,
            kernel_size ** 2 * self.groups,
            1, 1)

        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        # out = _involution_cuda(x, weight, stride=self.stride, padding=(self.kernel_size-1)//2)
        # print("weight shape:",weight.shape)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        # print("new out:",(weight*out).shape)
        out = (weight * out).sum(dim=3).view(b, self.c1, h, w)

        return out

class GhostConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = GhostBaseConv(c1, c_, k, s, None, g, act)
        self.cv2 = GhostBaseConv(c_, c_, 5, 1, None, c_, act)
        # self.cv1 = Conv(c1, c_, k, s, g, act)
        # self.cv2 = Conv(c_, c_, 5, 1, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)

class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# class Bottleneck(nn.Module):
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = GhostConv(c1, c_, k[0], 1)
#         self.cv2 = GhostConv(c_, c2, k[1], 1, g=g)
#         self.iv = Involution(c1,1,1)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         return self.iv(x) + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e) 
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):

        y = list(self.cv1(x).split((self.c, self.c), 1))

        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

        # carafe
        # from mmcv.ops import CARAFEPack
        # self.upsample = nn.Sequential(
        #     BasicConv(in_channels, out_channels, 1),
        #     CARAFEPack(out_channels, scale_factor=scale_factor)
        # )

    def forward(self, x):
        x = self.upsample(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        x = self.downsample(x)

        return x


class ASFF_2_1(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[512, 640]):
        super(ASFF_2_1, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(channel[1], compress_c, 1, 1)
        self.weight_level_2 = Conv(channel[0], compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(channel[0], channel[1], 3, 1)
        self.upsample = Upsample(channel[1], channel[0])
        self.downsample = Downsample(channel[0], channel[1])
        self.level = level

    def forward(self, x, y):
        input1 = x
        input2 = y
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input2 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out

class ASFF_2_2(nn.Module):
    def __init__(self, inter_dim=512, level=1, channel=[512, 640]):
        super(ASFF_2_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(channel[0], compress_c, 1, 1)
        self.weight_level_2 = Conv(channel[1], compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(channel[0], channel[1], 3, 1)
        self.upsample = Upsample(channel[1], channel[0])
        self.downsample = Downsample(channel[1], channel[0])
        self.level = level

    def forward(self, x, y):
        input1 = x
        input2 = y
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out

class ASFF_2_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[320, 640]):
        super(ASFF_2_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(channel[0], compress_c, 1, 1)
        self.weight_level_2 = Conv(channel[0], compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(channel[0], channel[0], 3, 1)
        self.upsample = Upsample(channel[1], channel[0])
        self.downsample = Downsample(channel[0], channel[1])
        self.level = level

    def forward(self, x, y):
        input1 = x
        input2 = y
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input2 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out


class Backbone(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()


        self.stem = Conv(3, base_channels, 3, 2)
        

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )

        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )

        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )

        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)

        x = self.dark3(x)
        feat1 = x

        x = self.dark4(x)
        feat2 = x

        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3
