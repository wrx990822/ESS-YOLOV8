import numpy as np
import torch
import torch.nn as nn

from .backbone import Backbone, C2f, Conv,ASFF_2_1,ASFF_2_2,ASFF_2_3
from .yolo_training import weights_init
from utils.utils_bbox import make_anchors
from .attetion import LCT


def fuse_conv_and_bn(conv, bn):

    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)


    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))


    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class DFL(nn.Module):

    def __init__(self, c1=16):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1

    def forward(self, x):

        b, c, a = x.shape

        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

        
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
        width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3

        self.backbone   = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)


        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")


        self.conv3_for_upsample1    = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.conv3_for_upsample2    = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)


        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)

        self.conv3_for_downsample1  = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)


        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)

        self.conv3_for_downsample2  = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul), base_depth, shortcut=False)


        ch              = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape      = None
        self.nl         = len(ch)
        # self.stride     = torch.zeros(self.nl)
        self.stride     = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        self.reg_max    = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no         = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes

        c2, c3   = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        if not pretrained:
            weights_init(self)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        return self

    def forward(self, x):

        feat1, feat2, feat3 = self.backbone.forward(x)


        P5_upsample = self.upsample(feat3)

        P4          = torch.cat([P5_upsample, feat2], 1)

        P4          = self.conv3_for_upsample1(P4)


        P4_upsample = self.upsample(P4)

        P3          = torch.cat([P4_upsample, feat1], 1)

        P3          = self.conv3_for_upsample2(P3)


        P3_downsample = self.down_sample1(P3)

        P4 = torch.cat([P3_downsample, P4], 1)

        P4 = self.conv3_for_downsample1(P4)


        P4_downsample = self.down_sample2(P4)

        P5 = torch.cat([P4_downsample, feat3], 1)

        P5 = self.conv3_for_downsample2(P5)

        shape = P3.shape


        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape


        box, cls        = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes), 1)

        dbox            = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)


# class YoloBody(nn.Module):
#     def __init__(self, input_shape, num_classes, phi, pretrained=False):
#         super(YoloBody, self).__init__()
#         depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
#         width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
#         deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
#         dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
#
#         base_channels = int(wid_mul * 64)  # 64
#         base_depth = max(round(dep_mul * 3), 1)  # 3
#
#         self.backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
#
#
#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
#
#
#         self.conv3_for_upsample1 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8,
#                                        base_depth, shortcut=False)
#
#         self.conv3_for_upsample2 = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth,
#                                        shortcut=False)
#
#
#         self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
#
#         self.conv3_for_downsample1 = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth,
#                                          shortcut=False)
#
#         # 512, 40, 40 => 512, 20, 20
#         self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
#         # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
#         self.conv3_for_downsample2 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8,
#                                          int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
#
#
#         ch = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
#         self.shape = None
#         self.nl = len(ch)
#
#         self.stride = torch.tensor(
#             [256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
#         self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
#         self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
#         self.num_classes = num_classes
#
#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
#         self.cv2 = nn.ModuleList(
#             nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
#         self.cv3 = nn.ModuleList(
#             nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
#         if not pretrained:
#             weights_init(self)
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
#
#         self.ASFF_2_1 = ASFF_2_1(level=0)
#         self.ASFF_2_2 = ASFF_2_2(level=1)
#         self.ASFF_2_3 = ASFF_2_3(level=0)
#         # self.LCT1 = LCT(640,8)
#         # self.LCT2 = LCT(512, 8)
#
#
#
#     def fuse(self):
#         print('Fusing layers... ')
#         for m in self.modules():
#             if type(m) is Conv and hasattr(m, 'bn'):
#                 m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
#                 delattr(m, 'bn')  # remove batchnorm
#                 m.forward = m.forward_fuse  # update forward
#         return self
#
#     def forward(self, x):
#
#         feat1, feat2, feat3 = self.backbone.forward(x)
#
#
#
#         P4_ASFF = self.ASFF_2_1(feat2,feat3)
#         #P4_ASFF = self.LCT1(P4_ASFF)
#
#         P5_ASFF = self.ASFF_2_2(feat2, feat3)
#        #P5_ASFF = self.LCT1(P5_ASFF)
#
#
#         P5_upsample = self.upsample(P5_ASFF)
#
#         P4 = torch.cat([P5_upsample, P4_ASFF], 1)
#
#         P4 = self.conv3_for_upsample1(P4)
#
#
#         P4_upsample = self.upsample(P4)
#
#         P3 = torch.cat([P4_upsample, feat1], 1)
#
#         P3 = self.conv3_for_upsample2(P3)
#
#
#         P3_ASFF = self.ASFF_2_3(P3, P4)
#
#         P3_downsample = self.down_sample1(P3_ASFF)
#
#         P4 = torch.cat([P3_downsample, P4], 1)
#
#         P4 = self.conv3_for_downsample1(P4)
#
#
#         P4_downsample = self.down_sample2(P4)
#
#         P5 = torch.cat([P4_downsample, feat3], 1)
#
#         P5 = self.conv3_for_downsample2(P5)
#
#         shape = P3.shape
#
#
#         x = [P3, P4, P5]
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#
#         if self.shape != shape:
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape
#
#
#         box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
#             (self.reg_max * 4, self.num_classes), 1)
#         # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
#         dbox = self.dfl(box)
#         return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)