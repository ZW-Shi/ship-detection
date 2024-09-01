# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:53:29 2021

@author: 1
"""
import torch
import torch.nn as nn

# DANet-空间注意力机制
class PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out

# DANet-通道注意力 
class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

## CBAM-空间注意力机制
class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1_2d = nn.Conv2d(2, 1, kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        x_avg_out = torch.mean(x, dim=1, keepdim=True)
        x_max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([x_avg_out, x_max_out], dim=1)
        cabm_feature = self.conv1_2d(concat)
        cabm_feature = self.sigmoid(cabm_feature)

        foaion = torch.mul(x, cabm_feature)
        return foaion
    
# CBAM-通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        cbam_channle = self.sigmoid(out)
        foaion = torch.mul(x, cbam_channle)
        return foaion

# 门控注意力机制
class CrossAttention(nn.Module):
    def __init__(self, in_channel=256, ratio=8):
        super(CrossAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channel, in_channel//ratio, kernel_size=1)
        self.conv_key   = nn.Conv2d(in_channel, in_channel//ratio, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        bz, c, h, w = x.shape
        depth_q = self.conv_query(x).view(bz, -1, h*w).permute(0, 2, 1)
        depth_k = self.conv_key(x).view(bz, -1, h*w)
        mask  = torch.bmm(depth_q, depth_k) # bz, hw, hw
        mask  = torch.softmax(mask, dim=-1)
        rgb_v = self.conv_value(x).view(bz, c, -1)
        feat  = torch.bmm(rgb_v, mask.permute(0, 2, 1)) # bz, c, hw
        feat  = feat.contiguous().view(bz, c, h, w)
        feat = self.gamma * feat + feat

        return feat

# SE-通道注意力机制    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
