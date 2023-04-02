import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights

from models.misc import nested_tensor_from_tensor_list
from models.misc import NestedTensor

MAX_SIZE = 256

class BackBone(nn.Module):

    def __init__(self):
        super(BackBone, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layers = list(self.model.children())[:-3]

    def forward(self, x):
        
        y = x.tensors
        assert y.shape[2:] == (MAX_SIZE, 2*MAX_SIZE)

        left = y[..., 0:MAX_SIZE]
        right = y[..., MAX_SIZE : 2*MAX_SIZE]
        
        for idx, layer in enumerate(self.layers):
            left = layer(left)
            right = layer(right)

        img = torch.cat([left, right], dim=-1)

        m = x.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=img.shape[-2:]).to(torch.bool)[0]

        out = NestedTensor(img,mask)
        
        return out



# ------TEST SECTION------
def test():
    print('Backbone test: ...')
    
    model = BackBone()

    img = torch.randn((1,3,256,512))
    img = nested_tensor_from_tensor_list(img)   

    model(img)

    breakpoint()
    print('Test Complete!')

if __name__ == '__main__':
    test()
    