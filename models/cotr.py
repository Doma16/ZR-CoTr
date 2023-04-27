import torch
import torch.nn as nn

from models.backbone import BackBone
from models.tranformer import Transformer
from models.pos_embed import PositionalEmbedding, NerfPositionalEncoding
from models.mlp import MLP

from models.misc import nested_tensor_from_tensor_list

class COTR(nn.Module):
    
    def __init__(self):
        super().__init__()
        #transformer
        self.backbone = BackBone()
        self.transformer = Transformer(return_intermediate=True, dropout=0.1)
        self.pos_emb = PositionalEmbedding(128) #from pos_embed

        hidden_dim = self.transformer.emb_dim
        
        self.mlp = MLP(hidden_dim, hidden_dim, 2, 3)

        self.proj_q = NerfPositionalEncoding(hidden_dim // 4)
        self.proj_x = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
    def forward(self, x, queries):    
        #to NestedTensor
        tensor = nested_tensor_from_tensor_list(x)
        
        #backbone
        y = self.backbone(tensor)
        #positional
        pos = self.pos_emb(y)

        y, mask = y.decompose()
        assert mask is not None
        #embedding queries
        b, q, _ = queries.shape
        queries = queries.reshape(-1,2)
        queries = self.proj_q(queries).reshape(b,q,-1)
        queries = queries.permute(1,0,2)
        
        #additional embedding after backbone for img
        y = self.proj_x(y)
        
        #transformer
        y = self.transformer(y, mask, queries, pos)[0]
        
        #MLP
        out_corr = self.mlp(y)        
        out = {'pred_corrs': out_corr[-1]}

        return out


#--------TEST--------
def test():
    
    cotr = COTR()
    
    img = torch.ones((1,3,256,512))
    queries = torch.randn((1,4,2))
    
    y = cotr(img,queries)
    
    breakpoint()

if __name__=='__main__':
    test()