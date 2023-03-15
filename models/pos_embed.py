import torch
import torch.nn as nn

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.sine = NerfPositionalEncoding(num_pos_feats // 2)

    
    @torch.no_grad()
    def forward(self, x):
        pass


class NerfPositionalEncoding(nn.Module):
    def __init__(self, depth = 32):
        super().__init__()
        self.bases = [i+1 for i in range(depth)]

    @torch.no_grad()
    def forward(self, x):
        return torch.cat([torch.sin(i * math.pi * x) for i in self.bases] + [torch.cos(i * math.pi * x) for i in self.bases], axis=-1)
    
