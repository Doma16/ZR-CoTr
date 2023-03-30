import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import KittiDataset
from utils import get_query

from torchvision.utils import save_image
from torchvision.utils import make_grid

from models.cotr import COTR

import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 1
LR = 1e-4
LR_BB = 0

# for Adam: beta1 = 0.9, beta2 = 0.98 , smallE = 10e-9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#still testing on cpu !
device = torch.device('cpu')

dataset = KittiDataset(root='../dataset')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

torch.cuda.empty_cache()
model = COTR()

opt_list = [{'params': model.transformer.parameters(), 'lr': LR},
            {'params': model.mlp.parameters(), 'lr': LR},
            {'params': model.pos_emb.parameters(), 'lr': LR},
            {'params': model.proj_q.parameters(), 'lr': LR}]

# backbone train?
if LR_BB > 0:
    opt_list.append({'params': model.backbone.parameters(), 'lr':LR_BB})

opt = optim.Adam(opt_list)

for batchid, (img1, img2, corrs) in enumerate(loader):
    query = corrs[:, 0, :, :]
    targets = corrs[:, 1, :, :]

    

    print(batchid)
    breakpoint()

    #plt.imshow(preview)
    #print(preview.shape)

breakpoint()