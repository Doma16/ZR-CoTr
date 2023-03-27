from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as t
import torchvision.transforms.functional as TF
import torch

import numpy as np
from PIL import Image


class KittiDataset(Dataset):
    def __init__(self, root, split='train', transforms = None, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.root = root
        self.split = split
        self.transforms = transforms
        self.ds = torchvision.datasets.Kitti2015Stereo(root=root, split=split, transforms=self.kitti_transform)
        self.ds._has_built_in_disparity_mask = False


    def __len__(self):
        return len(self.ds) // 2
    
    def __getitem__(self, index):
        return self.ds.__getitem__(index)

    def kitti_transform(self, imgs, dmap, valid_masks):
        
        img1 = imgs[0]
        img2 = imgs[1]

        dmap1 = dmap[0]
        dmap2 = dmap[1]

        # SHOULD COLLECT QUERIES BEFORE RESIZING ?
        # ...
        '''
        ch ,size_y, size_x = dmap1.shape
        assert ch == 1
        indicies = np.array(dmap1 > 0)
        indicies = indicies.squeeze(0)

        query_points = []
        for i in range(size_x):
            for j in range(size_y):
                if indicies[j][i]:
                    query_points.append([(i *1. / size_x ,j* 1. / size_y), ((i - dmap1[0][j][i]) * 1. / size_x, j * 1. / size_y)])
        '''
        
        # RESIZING
        resize = t.Resize(size=(self.img_size, self.img_size))

        img1 = resize(img1)
        img2 = resize(img2)
        
        '''
        dmap1 = dmap1.reshape((dmap1.shape[1], dmap1.shape[2]))
        dmap1 = Image.fromarray(dmap1)
        dmap2 = dmap2.reshape((dmap2.shape[1], dmap2.shape[2]))
        dmap2 = Image.fromarray(dmap2)

        dmap1 = dmap1.resize((self.img_size, self.img_size), resample=Image.Resampling.BICUBIC)
        dmap2 = dmap2.resize((self.img_size, self.img_size), resample=Image.Resampling.BICUBIC)
        
        dmap1 = np.array(dmap1)
        dmap2 = np.array(dmap2)
        '''

        # TO TENSOR
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        
        imgs = (img1, img2)

        #dmap = (dmap1, dmap2)
    
        dmap = (dmap1,dmap2)

        return imgs, dmap, valid_masks

def test():

    ds = KittiDataset(root='../dataset/')

    ds[0]
    
    breakpoint()
    1
 
if __name__ == '__main__':
    test()
