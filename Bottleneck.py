import torch
import torch.nn  as nn

def preprocess(x):
    return x.squeeze(1)



class Bottleneck(nn.Module):
    def __init__(self, in_dim, interm_dim, expansion, stride):
        super(Bottleneck,self).__init__()
        
        self.bottleneck_layer = nn.Sequential (
            nn.Conv2d(in_dim, interm_dim, kernel_size=1, stride=1, padding=0, bias=False ), #conv1x1
            nn.BatchNorm2d(interm_dim), #BN
            nn.ReLU(), 
            nn.Conv2d(interm_dim, interm_dim, kernel_size=3, stride=stride, padding=1, bias=False), #conv3x3
            nn.BatchNorm2d(interm_dim), #BN
            nn.ReLU(), 
            nn.Conv2d(interm_dim,  interm_dim * expansion, kernel_size=1, stride=1, padding=0, bias=False ),#conv1x1
            nn.BatchNorm2d(interm_dim * expansion ),
        )

        if in_dim == interm_dim * expansion:
            self.identity = True
        else:   #TODO: rever, nao entendi muito bem essa parte
            self.identity = False
            projection_layer = []
            projection_layer.append(nn.Conv2d(in_dim, interm_dim*expansion, kernel_size=1, stride=stride, padding=0, bias=False ))
            projection_layer.append(nn.BatchNorm2d(interm_dim*expansion))
            # Only conv->BN and no ReLU
            # projection_layer.append(nn.ReLU())
            self.projection = nn.Sequential(*projection_layer)
    
    def forward(self, x):
        input_in = x
        x = self.bottleneck_layer(x)

        if self.identity:
            x += input_in
        else:
            x += self.projection(input_in)

        # final relu
        x = nn.ReLU()(x)
        
        return x
    
def test_Bottleneck():
    batch_size = 10
    input_tensor = torch.randn(batch_size, 1, 128, 128, 128)
    x = preprocess(input_tensor)
    model = Bottleneck(128,64,4,2)
    print(model(x).shape)
    del model


if __name__ == "__main__":
    test_Bottleneck()