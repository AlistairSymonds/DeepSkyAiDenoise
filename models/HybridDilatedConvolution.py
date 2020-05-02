import torch.nn as nn
import torch
import torch.nn.init as init
from utils import initialize_weights

#This is the code the HDC block from the replicated custom model

# "A Multiscale Image Denoising Algorithm Based On Dilated Residual Convolution
# Network"
#paper: https://arxiv.org/pdf/1812.09131.pdf
class HDConv(nn.Module):
    def __init__(self):
        super(HDConv, self).__init__()
        self.dconv1 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=64, out_channels=64, dilation=1, padding=1
        )
        self.bnorm1 = nn.BatchNorm2d(64)
        self.dconv2 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=64, out_channels=64, dilation=3, padding=3
        )
        self.bnorm2 = nn.BatchNorm2d(64)
        self.dconv3 = nn.Conv2d(
            kernel_size=(3, 3), in_channels=64, out_channels=64, dilation=5, padding=5
        )
        self.bnorm3 = nn.BatchNorm2d(64)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()

    def forward(self, input):

        x = self.dconv1(input)
        conv1_result = self.prelu1(self.bnorm1(x))

        x = self.dconv2(conv1_result)
        x = self.prelu2(self.bnorm2(x))

        x = self.dconv3(x)
        x = self.prelu3(self.bnorm3(x))

        output = x + conv1_result
        return output

    def _initialize_weights(self):
        for m in self.modules():
            initialize_weights(m)