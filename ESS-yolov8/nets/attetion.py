
import torch
import torch.nn as nn
import math



class SE(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.silu1 = nn.SiLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.silu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.silu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x



class ECA(nn.Sequential):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.avg_pool(x)
        b = self.max_pool(x)
        y = a + b
        #y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#
#         mip = max(8, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#
#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#         # a_h = a_h.expand(-1, -1, h, w)
#         # a_w = a_w.expand(-1, -1, h, w)
#
#         out = identity * a_w * a_h
#
#         return out
###   Criss-Cross Attention

# class Involution(nn.Module):
#     def __init__(self, channels, kernel_size=1, stride=1, group_channels=16, reduction_ratio=4):
#         super().__init__()
#         #assert not (channels % group_channels or channels % reduction_ratio)
#
#         # in_c=out_c
#         self.channels = channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#
#         # 每组多少个通道
#         self.group_channels = group_channels
#         self.groups = channels // group_channels
#
#         # reduce channels
#         self.reduce = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction_ratio, 1),
#             nn.BatchNorm2d(channels // reduction_ratio),
#             nn.ReLU()
#         )
#         # span channels
#         self.span = nn.Conv2d(
#             channels // reduction_ratio,
#             self.groups * kernel_size ** 2,
#             1
#         )
#
#         self.down_sample = nn.AvgPool2d(stride) if stride != 1 else nn.Identity()
#         self.unfold = nn.Unfold(kernel_size, padding=(kernel_size - 1) // 2, stride=stride)
#
#     def forward(self, x):
#         # Note that 'h', 'w' are height & width of the output feature.
#
#         # generate involution kernel: (b,G*K*K,h,w)
#         weight_matrix = self.span(self.reduce(self.down_sample(x)))
#         b, _, h, w = weight_matrix.shape
#
#         # unfold input: (b,C*K*K,h,w)
#         x_unfolded = self.unfold(x)
#         # (b,C*K*K,h,w)->(b,G,C//G,K*K,h,w)
#         x_unfolded = x_unfolded.view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
#
#         # (b,G*K*K,h,w) -> (b,G,1,K*K,h,w)
#         weight_matrix = weight_matrix.view(b, self.groups, 1, self.kernel_size ** 2, h, w)
#         # (b,G,C//G,h,w)
#         mul_add = (weight_matrix * x_unfolded).sum(dim=3)
#         # (b,C,h,w)
#         out = mul_add.view(b, self.channels, h, w)
#
#         return out

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

def INF(B,H,W):

    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

# class CrissCrossAttention(nn.Module):
#     """ Criss-Cross Attention Module"""
#     def __init__(self, in_ch,ratio=8):
#         super(CrissCrossAttention,self).__init__()
#         self.q = nn.Conv2d(in_ch, in_ch//ratio, 1)
#         self.k = nn.Conv2d(in_ch, in_ch//ratio, 1)
#         self.v = nn.Conv2d(in_ch, in_ch, 1)
#         self.softmax = nn.Softmax(dim=3)
#         self.INF = INF
#         self.gamma = nn.Parameter(torch.zeros(1)) # 初始化为0
#
#
#     def forward(self, x):
#         bs, _, h, w = x.size()
#         # Q
#         x_q = self.q(x)
#         # b,c',h,w -> b,w,c',h -> b*w,c',h -> b*w,h,c'
#
#         x_q_H = x_q.permute(0,3,1,2).contiguous().view(bs*w,-1,h).permute(0, 2, 1)
#         # b,c',h,w -> b,h,c',w -> b*h,c',w -> b*h,w,c'
#         x_q_W = x_q.permute(0,2,1,3).contiguous().view(bs*h,-1,w).permute(0, 2, 1)
#         # K
#         x_k = self.k(x) # b,c',h,w
#         # b,c',h,w -> b,w,c',h -> b*w,c',h
#         x_k_H = x_k.permute(0,3,1,2).contiguous().view(bs*w,-1,h)
#         # b,c',h,w -> b,h,c',w -> b*h,c',w
#         x_k_W = x_k.permute(0,2,1,3).contiguous().view(bs*h,-1,w)
#         # V
#         x_v = self.v(x)
#         # b,c,h,w -> b,w,c,h -> b*w,c,h
#         x_v_H = x_v.permute(0,3,1,2).contiguous().view(bs*w,-1,h)
#         # b,c,h,w -> b,h,c,w -> b*h,c,w
#         x_v_W = x_v.permute(0,2,1,3).contiguous().view(bs*h,-1,w)
#
#         energy_H = (torch.bmm(x_q_H, x_k_H)+self.INF(bs, h, w)).view(bs,w,h,h).permute(0,2,1,3) # b,h,w,h
#
#         energy_W = torch.bmm(x_q_W, x_k_W).view(bs,h,w,w)
#
#         concate = self.softmax(torch.cat([energy_H, energy_W], 3)) # b,h,w,h+w
#
#
#         att_H = concate[:,:,:,0:h].permute(0,2,1,3).contiguous().view(bs*w,h,h)
#         #print(concate)
#         #print(att_H)
#         att_W = concate[:,:,:,h:h+w].contiguous().view(bs*h,w,w)
#
#         out_H = torch.bmm(x_v_H, att_H.permute(0, 2, 1)).view(bs,w,-1,h).permute(0,2,3,1) # b,c,h,w
#         out_W = torch.bmm(x_v_W, att_W.permute(0, 2, 1)).view(bs,h,-1,w).permute(0,2,1,3) # b,c,h,w
#         #print(out_H.size(),out_W.size())
#         return self.gamma*(out_H + out_W) + x

class CrissCrossAttention(nn.Module):

    def __init__(self, in_ch,ratio=8):
        super(CrissCrossAttention,self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch//ratio, 1)
        self.q = Involution(c1=in_ch//ratio,kernel_size=1, stride=1)
        self.k = Involution(c1=in_ch//ratio,kernel_size=1, stride=1)
        self.v = Involution(c1=in_ch,kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1)) # 初始化为0


    def forward(self, x):
        bs, _, h, w = x.size()
        # Q
        x_q = self.q(self.conv(x))

        x_q_H = x_q.permute(0,3,1,2).contiguous().view(bs*w,-1,h).permute(0, 2, 1)
        # b,c',h,w -> b,h,c',w -> b*h,c',w -> b*h,w,c'
        x_q_W = x_q.permute(0,2,1,3).contiguous().view(bs*h,-1,w).permute(0, 2, 1)
        # K
        x_k = self.k(self.conv(x)) # b,c',h,w
        # b,c',h,w -> b,w,c',h -> b*w,c',h
        x_k_H = x_k.permute(0,3,1,2).contiguous().view(bs*w,-1,h)
        # b,c',h,w -> b,h,c',w -> b*h,c',w
        x_k_W = x_k.permute(0,2,1,3).contiguous().view(bs*h,-1,w)
        # V
        x_v = self.v(x)
        # b,c,h,w -> b,w,c,h -> b*w,c,h
        x_v_H = x_v.permute(0,3,1,2).contiguous().view(bs*w,-1,h)
        # b,c,h,w -> b,h,c,w -> b*h,c,w
        x_v_W = x_v.permute(0,2,1,3).contiguous().view(bs*h,-1,w)

        energy_H = (torch.bmm(x_q_H, x_k_H)+self.INF(bs, h, w)).view(bs,w,h,h).permute(0,2,1,3) # b,h,w,h

        energy_W = torch.bmm(x_q_W, x_k_W).view(bs,h,w,w)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3)) # b,h,w,h+w


        att_H = concate[:,:,:,0:h].permute(0,2,1,3).contiguous().view(bs*w,h,h)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,h:h+w].contiguous().view(bs*h,w,w)

        out_H = torch.bmm(x_v_H, att_H.permute(0, 2, 1)).view(bs,w,-1,h).permute(0,2,3,1) # b,c,h,w
        out_W = torch.bmm(x_v_W, att_W.permute(0, 2, 1)).view(bs,h,-1,w).permute(0,2,1,3) # b,c,h,w
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=8):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(64, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.involution = Involution(mip, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Sequential(
                      Involution(mip, kernel_size=7, stride=1),
                      nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0))
        self.conv_w = nn.Sequential(
                      Involution(mip, kernel_size=7, stride=1),
                      nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.involution(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        # a_h = a_h.expand(-1, -1, h, w)
        # a_w = a_w.expand(-1, -1, h, w)

        out = identity * a_w * a_h

        return out


class LCT(nn.Module):
    def __init__(self, channels, groups, eps=1e-5):
        super().__init__()
        assert channels % groups == 0, "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.w = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        y = self.avgpool(x).view(batch_size, self.groups, -1)
        mean = y.mean(dim=-1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=-1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_norm = y_norm.reshape(batch_size, self.channels, 1, 1)
        y_norm = self.w.reshape(1, -1, 1, 1) * y_norm + self.b.reshape(1, -1, 1, 1)
        y_norm = self.sigmoid(y_norm)
        return x * y_norm.expand_as(x)