import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from models.backbone import BackBone
from dataset import KittiDataset
from utils import kitti_transform

BATCH_SIZE = 8

# for Adam: beta1 = 0.9, beta2 = 0.98 , smallE = 10e-9

backboneEncoder = BackBone().eval()

dataset = KittiDataset(root='../dataset', transforms=kitti_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#need transforms

for batchid, (img1, img2, dmap) in enumerate(loader):
    
    print(batchid)



    breakpoint()
    1

breakpoint()