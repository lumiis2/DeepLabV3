import torch
import torch.nn  as nn

class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, expansion, stride):
        super(Bottleneck,self).__init__()
        
        self.activation = nn.ReLU()

        if in_dim == out_dim * expansion:
            self.identity = True
        else:   #TODO: rever, nao entendi muito bem essa parte
            self.identity = False
            projection_layers = [
                nn.Conv3d(in_channels=in_dim, out_channels=out_dim * expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm3d(out_dim * expansion)
            ]
            self.projection = nn.Sequential(*projection_layers)
        
        self.bottleneck_layer = nn.Sequential (
            nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False ), #conv1x1
            nn.BatchNorm3d(out_dim), #BN
            self.activation, 
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False), #conv3x3
            nn.BatchNorm3d(out_dim), #BN
            self.activation, 
            nn.Conv3d(out_dim,  out_dim * expansion, kernel_size=1, stride=1, padding=0, bias=False ),#conv1x1
            nn.BatchNorm3d(out_dim * expansion ),
        )
    
    def forward(self, x):
        input_in = x
        x = self.bottleneck_layer(x)

        if self.identity:
            x += input_in
        else:
            x += self.projection(input_in)

        # final relu
        x = self.activation(x)
        
        return x
    