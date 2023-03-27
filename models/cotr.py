import torch
import torch.nn as nn

from .backbone import BackBone
from .tranformer import Transformer
from .pos_embed import NerfPositionalEncoding
from .mlp import MLP

class COTR(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.backbone = BackBone()
        self.transformer = Transformer()
        self.pos_emb = NerfPositionalEncoding()
        hidden_dim = self.transformer.emb_dim
        self.mlp = MLP(hidden_dim, hidden_dim, 2, 3)
        
    def forward(self, x, queries):
        
        y = self.backbone(x)
        
        breakpoint()