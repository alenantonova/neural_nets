import torch
import torch.nn as nn


def double_conv_bn_relu(in_planes, out_planes, kernel=3, stride=1, padding=1):
    net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(num_features=out_planes),
            nn.ReLU(True)
        )
    
    return net


class convDown(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(convDown, self).__init__()
        self.conv_down = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv_bn_relu(in_planes, out_planes))

    def forward(self, input):
        input = self.conv_down(input)
        return input


class convUp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(convUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_planes//2, in_planes//2, 2, stride=2)
        self.conv = double_conv_bn_relu(in_planes, out_planes)

    def forward(self, input1, input2):
        input1 = self.up(input1)
        x = torch.cat([input2, input1], dim=1)
        x = self.conv(x)
        return x
    

class SegmenterModel(nn.Module):
    def __init__(self, in_size=3):
        super(SegmenterModel, self).__init__()
        self.first_conv = double_conv_bn_relu(in_size, 64)
        
        self.conv_down1 = convDown(64, 128)
        self.conv_down2 = convDown(128, 256)
        self.conv_down3 = convDown(256, 512)
        self.conv_down4 = convDown(512, 512)
        
        self.conv_up1 = convUp(1024, 256)
        self.conv_up2 = convUp(512, 128)
        self.conv_up3 = convUp(256, 64)
        self.conv_up4 = convUp(128, 64)
        
        self.last_conv = nn.Conv2d(64, 1, 1)

    def forward(self, input):
        x1 = self.first_conv(input)
        
        x2 = self.conv_down1(x1)
        x3 = self.conv_down2(x2)
        x4 = self.conv_down3(x3)
        x5 = self.conv_down4(x4)
        
        input = self.conv_up1(x5, x4)
        input = self.conv_up2(input, x3)
        input = self.conv_up3(input, x2)
        input = self.conv_up4(input, x1)
        
        input = self.last_conv(input)
        return input