import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, num_channel):
        super(ASPP, self).__init__()
            # 512 corresponds to the number of filters at the output of 
        self.conv_1x1_1 = nn.Conv3d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm3d(256)

        self.conv_3x3_1 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm3d(256)

        self.conv_3x3_2 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm3d(256)

        self.conv_3x3_3 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm3d(256)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv_1x1_2 = nn.Conv3d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm3d(256)

        self.conv_1x1_3 = nn.Conv3d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm3d(256)

        self.conv_1x1_4 = nn.Conv3d(256, num_channel, kernel_size=1)

    def forward(self, feature_map):
            feature_map_h = feature_map.size()[2]
            feature_map_w = feature_map.size()[3]
            feature_map_c = feature_map.size()[4]

            # Convolution
            out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
            # Convolution dilated
            out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
            out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
            out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))
            # Sharing
            out_img = self.avg_pool(feature_map)
            out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
            # oversampling to find the size of the feature map
            out_img = F.interpolate(out_img, size=(feature_map_h,
            feature_map_w, feature_map_c), mode='trilinear', align_corners=True)

            # Concatenation of the 5 operations performed in parallel
            out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
            out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

            out = self.conv_1x1_4(out)

            return out

def test_aspp():
    in_channels = 512 
    out_channels = 64  
    input_tensor = torch.randn(1, in_channels, 28, 28, 28)  

    aspp = ASPP(in_channels)

    output = aspp(input_tensor)

    print("Output shape:", output.shape)  

test_aspp()