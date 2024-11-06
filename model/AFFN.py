import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(MultiScaleFusion, self).__init__()
        self.scale1 = ConvBNReLU(in_channels, mid_channels, kernel_size=1)
        self.scale2 = ConvBNReLU(in_channels, mid_channels, kernel_size=3, padding=1)
        self.scale3 = ConvBNReLU(in_channels, mid_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x1 = self.scale1(x)
        x2 = self.scale2(x)
        x3 = self.scale3(x)
        return x1 + x2 + x3

class AFFN(nn.Module):
    def __init__(self, in_channels_local, in_channels_global, mid_channels):
        super(AFFN, self).__init__()

        # Local branch multi-scale fusion with depthwise separable convolution
        self.local_scale1 = ConvBNReLU(in_channels_local, mid_channels, kernel_size=1)
        self.local_scale2 = DepthwiseSeparableConv(in_channels_local, mid_channels, kernel_size=3, padding=1)
        self.local_scale3 = DepthwiseSeparableConv(in_channels_local, mid_channels, kernel_size=5, padding=2)

        # Global branch multi-scale fusion with depthwise separable convolution
        self.global_scale1 = ConvBNReLU(in_channels_global, mid_channels, kernel_size=1)
        self.global_scale2 = DepthwiseSeparableConv(in_channels_global, mid_channels, kernel_size=3, padding=1)
        self.global_scale3 = DepthwiseSeparableConv(in_channels_global, mid_channels, kernel_size=5, padding=2)

        self.channel_attention = ChannelAttention(mid_channels)
        self.spatial_attention = SpatialAttention()

        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels * 2, mid_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels // 16, 2, 1),
            nn.Softmax(dim=1)
        )

        self.fusion_conv = ConvBNReLU(mid_channels, mid_channels, 1)
        self.residual_conv = ConvBNReLU(mid_channels, mid_channels, 1)

    def forward(self, x, y):
        # Local branch multi-scale fusion
        x1 = self.local_scale1(x)
        x2 = self.local_scale2(x)
        x3 = self.local_scale3(x)
        x = x1 + x2 + x3

        # Global branch multi-scale fusion
        y1 = self.global_scale1(y)
        y2 = self.global_scale2(y)
        y3 = self.global_scale3(y)
        y = y1 + y2 + y3

        y = F.interpolate(y, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=False)

        # Apply Channel Attention
        x = self.channel_attention(x) * x
        y = self.spatial_attention(y) * y

        # Predict weights
        combined = torch.cat([x, y], dim=1)
        weights = self.weight_predictor(combined)

        # Ensure weights are reshaped correctly
        local_weight = weights[:, 0, :, :].unsqueeze(1)
        global_weight = weights[:, 1, :, :].unsqueeze(1)

        # Weighted sum of local and global features
        combined = local_weight * x + global_weight * y
        # combined = self.spatial_attention(combined) * combined

        # Fusion
        fused_features = self.fusion_conv(combined)
        fused_features = fused_features + self.residual_conv(x) + self.residual_conv(y)
        return fused_features

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)