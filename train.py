import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from models.backbone import BackBone
from dataset import KittiDataset

BATCH_SIZE = 8

backboneEncoder = BackBone().eval()

dataset = KittiDataset(root='../dataset')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for batchid, (img1, img2, dmap) in enumerate(loader):
    
    print(batchid)

    

    breakpoint()

breakpoint()