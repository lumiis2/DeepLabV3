import torch
import torch.nn  as nn

class AtrousLayer3D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation):
        super(AtrousLayer3D, self).__init__()
        self.atrous_conv = nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        return self.atrous_conv(x)