#RESNET50 BACKBONE - (shallow features)
# SOURCES
# https://www.kaggle.com/code/gxkok21/resnet50-with-pytorch
# https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758
# https://medium.com/@karuneshu21/how-to-resnet-in-pytorch-9acb01f36cf5


from BottleneckLayer import Bottleneck
from AtrousLayer import AtrousLayer3D
import torch
import torch.nn as nn

# resnetX = (Num of channels (de cada saida), repetition, Bottleneck_expansion)
# model_parameters = ([64,128,256,512],[3,4,6,3],4)    
# Resnet + Atrous Layer

class Resnet50(nn.Module):
    """
        Creates the ResNet50 architecture based on the provided variant.
        Args:
            model_parameters (list) : eg. [[64,128,256,512],[3,4,6,3],4,True]
            in_dim (int) : image channels (3)
            num_classes (int) : output #classes 

    """
    def __init__(self, model_parameters, in_dim, num_classes):
        
        super(Resnet50, self).__init__()
        self.dim_list = model_parameters[0]
        self.repeatition_list = model_parameters[1]
        self.expansion = model_parameters[2]
        self.activation = nn.ReLU()
        self.atrous_layer = AtrousLayer3D(2048, num_classes, kernel_size=(2, 2, 2), dilation=(2, 2, 2))  #rate = 2 no paper

        self.first_block = nn.Sequential (
            nn.Conv3d(in_channels=in_dim, out_channels=64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            self.activation,
        )
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3),stride=(2, 2, 2),padding=(1, 1, 1))
        
        self.block1 = self._make_blocks( 64 , self.dim_list[0], self.repeatition_list[0], self.expansion, stride=1 )
        self.block2 = self._make_blocks( self.dim_list[0]*self.expansion , self.dim_list[1], self.repeatition_list[1], self.expansion, stride=2 )
        self.block3 = self._make_blocks( self.dim_list[1]*self.expansion , self.dim_list[2], self.repeatition_list[2], self.expansion, stride=2 )
        self.block4 = self._make_blocks( self.dim_list[2]*self.expansion , self.dim_list[3], self.repeatition_list[3], self.expansion, stride=2 )

        self.average_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear( self.dim_list[3]*self.expansion , num_classes)

    def _make_blocks(self, in_dim, out_dim, num_repeat, expansion, stride):

            layers = [] 
            layers.append(Bottleneck(in_dim, out_dim, expansion, stride=stride))
            for num in range(1,num_repeat):
                layers.append(Bottleneck(out_dim*expansion,out_dim,expansion, stride=1))

            return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.first_block(x)
        x = self.maxpool(x)
        
        x = self.block1(x)
        
        x = self.block2(x)
        
        x = self.block3(x)
        
        x = self.block4(x)
        print(x.shape)

        """x = self.average_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)"""

        x = self.atrous_layer(x)
        
        return x
    
def test_ResNet(params):
    batch_size = 1
    x = torch.randn(batch_size, 1, 128, 128, 128)
    model = Resnet50(params, in_dim=1, num_classes=1000)
    output = model(x)
    print(output.shape)
    return model

def test_Bottleneck():
    batch_size = 1
    x = torch.randn(batch_size, 64, 112, 112, 112)
    model = Bottleneck(64,64,4,2)
    print(model(x).shape)
    del model

def test_AtrousLayer3D():
    batch_size = 1
    atrous_layer = AtrousLayer3D(1, 10, (3, 3, 3), (2, 2, 2))
    input_data = torch.randn(batch_size, 1, 32, 32, 32)
    output_data = atrous_layer(input_data)

    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)


if __name__ == "__main__":
    model_parameters = ([64,128,256,512],[3,4,6,3],4)
    model = test_ResNet(model_parameters) # esse ta rolando
    test_Bottleneck() #TODO  tem algm erro nesse bglh
    test_AtrousLayer3D()

#ATROUS BACKBONE - (deeper features)

#ASPP

#HEAD





