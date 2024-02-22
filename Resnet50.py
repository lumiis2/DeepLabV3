#RESNET50 BACKBONE - (shallow features)
# SOURCES
# https://www.kaggle.com/code/gxkok21/resnet50-with-pytorch
# https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758
# https://medium.com/@karuneshu21/how-to-resnet-in-pytorch-9acb01f36cf5


from Bottleneck import Bottleneck
import torch
import torch.nn  as nn

# resnetX = (Num of channels (de cada saida), repetition, Bottleneck_expansion)
# model_parameters = ([64,128,256,512],[3,4,6,3],4)    


class Resnet50(nn.Module):
    def __init__(self, model_parameters, in_dim, num_classes):
        super(Resnet50, self).__init__()
        self.dim_list = model_parameters[0]
        self.repeatition_list = model_parameters[1]
        self.expansion = model_parameters[2]
        self.activation = nn.ReLU()

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

    def _make_blocks(self, in_dim, intermediate_dim, num_repeat, expansion, stride):

            layers = [] 

            layers.append(Bottleneck(in_dim, intermediate_dim, expansion, stride=stride))
            for num in range(1,num_repeat):
                layers.append(Bottleneck(intermediate_dim*expansion,intermediate_dim,expansion, stride=1))

            return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.first_block(x)
        x = self.maxpool(x)
        
        x = self.block1(x)
        
        x = self.block2(x)
        
        x = self.block3(x)
        
        x = self.block4(x)
        
        x = self.average_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        
        return x
    
def test_ResNet(params):
    batch_size = 1
    x = torch.randn(batch_size, 1, 128, 128, 128)
    model = Resnet50(params, in_dim=1, num_classes=1000)
    output = model(x)
    print(output.shape)
    return model

def test_Bottleneck():
    batch_size = 10
    x = torch.randn(batch_size, 1, 128, 128, 128)
    # x = torch.randn(1,64,112,112)
    model = Bottleneck(1,64,4,2)
    print(model(x).shape)
    del model


if __name__ == "__main__":
    model_parameters = ([64,128,256,512],[3,4,6,3],4)
    model = test_ResNet(model_parameters) # esse ta rolando
    #test_Bottleneck() #TODO  tem algm erro nesse bglh


#ATROUS BACKBONE - (deeper features)

#ASPP

#HEAD





