import torch
import torch.nn as nn

from models.backbone import BackBone
from models.tranformer import Transformer
from models.pos_embed import PositionalEmbedding, NerfPositionalEncoding
from models.mlp import MLP

from models.misc import nested_tensor_from_tensor_list

# hyper_params:
# BackBone
# -no params (using ResNet18)
# Transformer
EMB_DIM = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
RETURN_INTERMEDIATE = True
DROPOUT = 0.1
# MLP
NLAYERS = 3
# other
# -

class COTR(nn.Module):
    
    def __init__(self,
                 emd_dim,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 return_intermediate,
                 dropout,
                 nlayers
                 ):
        super().__init__()
        #transformer
        self.backbone = BackBone()
        self.transformer = Transformer(
            emb_dim=emd_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            return_intermediate=return_intermediate,
            dropout=dropout
        )

        self.pos_emb = PositionalEmbedding(128) #from pos_embed

        hidden_dim = self.transformer.emb_dim
        
        self.mlp = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=2,
            nlayers=nlayers
        )

        self.proj_q = NerfPositionalEncoding(hidden_dim // 4)
        self.proj_x = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=1
        )
        
    def forward(self, x, queries):    
        #to NestedTensor
        if isinstance(x, (list, torch.Tensor)):
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