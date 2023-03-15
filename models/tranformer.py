import torch
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, nhead=8, 
                 num_encoder_layers=6,
                 num_decoder_layers=6):
        super(Transformer, self).__init__()

        self.encoder = 1

        self.decoder = 1


class TransformerEncoder(nn.Module):

    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.layers = 1
        self.nlayers = 1

class TransformerDecoder(nn.Module):
    
    def __init__(self):
        super(TransformerDecoder, self).__init__()
        self.layers = 1
        self.nlayers = 1

class T_E_Layer(nn.Module):
    
    def __init__(self, emb_dim, nhead, dim_forward=2048, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(emb_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(emb_dim, dim_forward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_forward, emb_dim)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def pos_embed(self, x, pos):
        return x if pos is None else x + pos

    def forward(self,x, pos):
        q = k = self.pos_embed(x, pos)

        x2 = self.attn(query=q,
                         key=k,
                         value=x)
        
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        return x
    
class T_D_Layer(nn.Module):
    
    def __init__(self, emb_dim, nhead, dim_forward=2048, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(emb_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(emb_dim, dim_forward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_forward, emb_dim)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def pos_embed(self, x, pos):
        return x if pos is None else x + pos

    def forward(self, x, memory, pos, query_pos):
        q = self.pos_embed(x, query_pos)
        k = self.pos_embed(memory, pos)
        x2 = self.attn(query=q,
                       key=k,
                       value=memory)
        
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
