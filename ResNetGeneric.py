import torch
import torch.nn as nn

"""
Source: https://medium.com/@karuneshu21/how-to-resnet-in-pytorch-9acb01f36cf5
The code has been edited and transformed to perform 3D convolutions and handle 3D data.
"""

class Bottleneck(nn.Module):
    """
        Constructs a Bottleneck block used in ResNet.

        Parameters:
            in_channels (int): Number of input channels.
            intermediate_channels (int): Number of intermediate channels.
            expansion (int): Expansion factor for the number of intermediate channels.
            is_Bottleneck (bool): If True, uses the Bottleneck version of the block, otherwise, uses the standard version.
            stride (int): Stride for convolution.

        Returns:
            nn.Module: Configured Bottleneck block.
    """
    
    def __init__(self, in_channels, intermediate_channels, expansion, is_Bottleneck, stride):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.is_Bottleneck = is_Bottleneck
        
        if self.in_channels == self.intermediate_channels * self.expansion:
            self.identity = True
        else:
            self.identity = False
            projection_layers = [
                nn.Conv3d(in_channels=self.in_channels, out_channels=self.intermediate_channels * self.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm3d(self.intermediate_channels * self.expansion)
            ]
            self.projection = nn.Sequential(*projection_layers)

        self.relu = nn.ReLU()

        if self.is_Bottleneck:
            self.conv1_1x1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.batchnorm1 = nn.BatchNorm3d(self.intermediate_channels)
            self.conv2_3x3 = nn.Conv3d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.batchnorm2 = nn.BatchNorm3d(self.intermediate_channels)
            self.conv3_1x1 = nn.Conv3d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
            self.batchnorm3 = nn.BatchNorm3d(self.intermediate_channels * self.expansion)
        else:
            self.conv1_3x3 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.batchnorm1 = nn.BatchNorm3d(self.intermediate_channels)
            self.conv2_3x3 = nn.Conv3d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.batchnorm2 = nn.BatchNorm3d(self.intermediate_channels)

    def forward(self, x):
        in_x = x

        if self.is_Bottleneck:
            x = self.relu(self.batchnorm1(self.conv1_1x1(x)))
            x = self.relu(self.batchnorm2(self.conv2_3x3(x)))
            x = self.batchnorm3(self.conv3_1x1(x))
        else:
            x = self.relu(self.batchnorm1(self.conv1_3x3(x)))
            x = self.batchnorm2(self.conv2_3x3(x))

        if self.identity:
            x += in_x
        else:
            x += self.projection(in_x)

        x = self.relu(x)
        
        return x


class ResNet(nn.Module):
    """
        Constructs a ResNet Neural Network.

        Parameters:
            resnet_variant (tuple): A tuple containing information about the ResNet architecture.
            in_channels (int): Number of input channels.
            num_classes (int): Number of classes in the output.

        Returns:
            nn.Module: Configured ResNet network.
    """
    def __init__(self, resnet_variant, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.channels_list = resnet_variant[0]
        self.repeatition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_Bottleneck = resnet_variant[3]

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.batchnorm1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.block1 = self._make_blocks(64, self.channels_list[0], self.repeatition_list[0], self.expansion, self.is_Bottleneck, stride=1)
        self.block2 = self._make_blocks(self.channels_list[0] * self.expansion, self.channels_list[1], self.repeatition_list[1], self.expansion, self.is_Bottleneck, stride=2)
        self.block3 = self._make_blocks(self.channels_list[1] * self.expansion, self.channels_list[2], self.repeatition_list[2], self.expansion, self.is_Bottleneck, stride=2)
        self.block4 = self._make_blocks(self.channels_list[2] * self.expansion, self.channels_list[3], self.repeatition_list[3], self.expansion, self.is_Bottleneck, stride=2)

        self.average_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(self.channels_list[3] * self.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.average_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        
        return x

    def _make_blocks(self, in_channels, intermediate_channels, num_repeat, expansion, is_Bottleneck, stride):
        """
        Creates blocks for building the ResNet network.

        Parameters:
            in_channels (int): Number of input channels.
            intermediate_channels (int): Number of intermediate channels.
            num_repeat (int): Number of repetitions of the block.
            expansion (int): Expansion factor for the number of intermediate channels.
            is_Bottleneck (bool): If True, uses the Bottleneck version of the block, otherwise, uses the standard version.
            stride (int): Stride for convolution.

        Returns:
            nn.Sequential: Configured sequential blocks.
        """
        layers = [] 
        layers.append(Bottleneck(in_channels, intermediate_channels, expansion, is_Bottleneck, stride=stride))
        for num in range(1, num_repeat):
            layers.append(Bottleneck(intermediate_channels * expansion, intermediate_channels, expansion, is_Bottleneck, stride=1))
        return nn.Sequential(*layers)


def test_ResNet(params):
    model = ResNet(params, in_channels=3, num_classes=1000)
    x = torch.randn(1, 3, 128, 128, 128)  # Example of 3D input
    output = model(x)
    print(output.shape)
    return model


if __name__ == "__main__":
    model_parameters = ([64, 128, 256, 512], [3, 4, 6, 3], 4, True)
    test_ResNet(model_parameters)