import torch
import torch.nn as nn
from models.HybridDilatedConvolution import HDConv
from utils import initialize_weights

#this model was replicating the work of group researchers from Chongqing University

# "A Multiscale Image Denoising Algorithm Based On Dilated Residual Convolution
# Network"
#paper: https://arxiv.org/pdf/1812.09131.pdf
class multiscale_denoise(nn.Module):
    def __init__(self):
        super(multiscale_denoise, self).__init__()
        self.multiscale7 = nn.Conv2d(
            in_channels=1, out_channels=12, kernel_size=(7, 7), padding=3
        )
        self.multiscale5 = nn.Conv2d(
            in_channels=1, out_channels=20, kernel_size=(5, 5), padding=2
        )
        self.multiscale3 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1
        )

        self.ms7_act = nn.PReLU()
        self.ms5_act = nn.PReLU()
        self.ms3_act = nn.PReLU()

        self.hdc1 = HDConv()
        self.hdc2 = HDConv()
        self.hdc3 = HDConv()

        self.final_dconv1 = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=(3, 3), dilation=1, padding=1
        )
        self.hdc_out_act = nn.PReLU()
        self.final_bnorm = nn.BatchNorm2d(num_features=1)
        self.final_activation = nn.PReLU()

        # we will use the externally initialised weights (Kaiming)
        # self._initialize_weights()

    def forward(self, x):

        ms_7x7_out = self.ms7_act(self.multiscale7(x))
        ms_5x5_out = self.ms5_act(self.multiscale5(x))
        ms_3x3_out = self.ms3_act(self.multiscale3(x))

        concat_ms = torch.cat([ms_7x7_out, ms_5x5_out, ms_3x3_out], dim=1)
        hdc1_out = self.hdc1(concat_ms)
        hdc2_out = self.hdc2(hdc1_out)
        hdc3_out = self.hdc_out_act(self.hdc3(hdc2_out))
        out = self.final_dconv1(hdc3_out)
        residual = self.final_bnorm(out)

        return x - residual

    def _initialize_weights(self):
        for m in self.modules():
            initialize_weights(m)