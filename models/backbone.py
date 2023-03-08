import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights

class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layers = list(self.model.children())[:-3]

    def forward(self, x):
        
        for idx, layer in enumerate(self.layers):
            x = layer(x)

        return x



# ------TEST SECTION------
def test():
    print('Backbone test: ...')
    
    model = BackBone()
    assert (model(torch.randn(1,3,256,256)).cpu().detach().numpy().shape) == (1,256,16,16)
    
    print('Test Complete!')

if __name__ == '__main__':
    test()
    