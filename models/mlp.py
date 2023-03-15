import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers):
        super().__init__()
        self.nlayers = nlayers
        h = [hidden_dim] * (nlayers - 1)
        self.layers = nn.ModuleList(nn.Linear(n,k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if i < self.nlayers - 1 else layer(x)
        
        return x

# --- Testing ---
def test():
    mlp = MLP(4,8,10,2)
    b = torch.randn((10,4))
    y = mlp(b)

if __name__ == '__main__':
    test()