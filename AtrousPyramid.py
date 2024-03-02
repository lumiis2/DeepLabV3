"""
estrutura do modelo:
1. PYRAMID
    menor-> maior
    1x1 conv
    3x3 conv rate = 6
    3x3 conv rate = 12
    3x3 conv arte = 18

2. AVG POLLING
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # 1x1 convolução
        self.conv_1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.conv_3x3_r6 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv_3x3_r12 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv_3x3_r18 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        conv_1x1_out = self.conv_1x1(x)
        conv_3x3_r6_out = self.conv_3x3_r6(x)
        conv_3x3_r12_out = self.conv_3x3_r12(x)
        conv_3x3_r18_out = self.conv_3x3_r18(x)

        output = torch.cat([conv_1x1_out, conv_3x3_r6_out, conv_3x3_r12_out, conv_3x3_r18_out], dim=1)

        output = self.avg_pool(output)

        return output

# o tamanho que sai da resnet apos a atrousbackbone eh Output shape: torch.Size([1, 10, 28, 28, 28])
#TODO: conferir paramentros
def test_aspp():
    in_channels = 10 
    out_channels = 64  
    input_tensor = torch.randn(1, in_channels, 28, 28, 28)  

    aspp = ASPP(in_channels, out_channels)

    output = aspp(input_tensor)

    print("Output shape:", output.shape)  

test_aspp()