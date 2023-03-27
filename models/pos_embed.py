import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import math

# for testing
from misc import nested_tensor_from_tensor_list
from misc import NestedTensor

class PositionalEmbedding(nn.Module):
    def __init__(self, num_pos_feats=64):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.sine = NerfPositionalEncoding(num_pos_feats // 2)

    
    @torch.no_grad()
    def forward(self, tl):
        x = tl.tensors
        mask = tl.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = (y_embed-0.5) / (y_embed[:,-1:, :] + eps)
        x_embed = (x_embed-0.5) / (x_embed[:, :, -1:] + eps)
        pos = torch.stack([x_embed, y_embed], dim=-1)
        return self.sine(pos).permute(0,3,1,2)


class NerfPositionalEncoding(nn.Module):
    def __init__(self, depth = 512):
        super().__init__()
        self.bases = [i+1 for i in range(depth)]

    @torch.no_grad()
    def forward(self, x):
        sin_v = [torch.sin(i * math.pi * x) for i in self.bases]
        #sin_v = [torch.tensor(i*x) for i in self.bases]
        cos_v = [torch.cos(i * math.pi * x) for i in self.bases]
        #cos_v = [torch.tensor(i*x) for i in self.bases]
        ret = torch.cat(sin_v + cos_v, axis=-1)
        return ret
    
def test():
    pe = PositionalEmbedding()

    img = torch.ones((1,1024,16,32))

    t = nested_tensor_from_tensor_list(img)

    y = pe(t)
    breakpoint()

if __name__ == '__main__':
    test()