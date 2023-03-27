import torch
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, emb_dim=128, nhead=8, 
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 return_intermediate=False):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(emb_dim, nhead, num_encoder_layers)
        self.decoder = TransformerDecoder(emb_dim, nhead, num_decoder_layers, return_intermediate)

        self.reset_parameters()
        
        self.emb_dim = emb_dim
        self.nhead = nhead

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                
    def forward(self, x, query, pos):
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2,0,1)
        pos = pos.flatten(2).permute(2,0,1)
        
        tgt = torch.zeros_like(query)
        memory = self.encoder(x, pos)
        hs = self.decoder(tgt, memory, pos, query)
        
        return hs.transpose(1,2), memory.permute(1,2,0).view(b,c,h,w)

class TransformerEncoder(nn.Module):

    def __init__(self, emb_dim, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList( T_E_Layer(emb_dim, nhead)  for i in range(num_layers) )
        self.num_layers = num_layers
        
    def forward(self, x, pos):
        y = x
        
        for layer in self.layers:
            y = layer(y, pos)
            
        return y

class TransformerDecoder(nn.Module):
    
    def __init__(self, emb_dim, nhead, num_layers, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList( T_D_Layer(emb_dim, nhead) for i in range(num_layers) )
        self.num_layers = num_layers

        self.return_intermediate = return_intermediate
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, memory, pos, query):
        y = x
        
        intermediate = []
        
        for layer in self.layers:
            y = layer(y, memory, pos, query)
            if self.return_intermediate:
                intermediate.append(self.norm(y))
                
        if self.norm is not None:
            y = self.norm(y)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(y)
                
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return y.unsqueeze(0)
            
            
        return y

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
