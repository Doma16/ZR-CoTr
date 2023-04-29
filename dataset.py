from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as tform
import torchvision.transforms.functional as TF
import torch

import numpy as np
from PIL import Image

import cv2
import matplotlib.pyplot as plt
from utils import two_images_side_by_side

class KittiDataset(Dataset):
    def __init__(self, root, split='train', transforms = None, img_size=256, num_kp=100):
        super().__init__()
        self.img_size = img_size
        self.root = root
        self.split = split
        self.num_kp = num_kp
        self.transforms = self.kitti_transform_train if split == 'train' else self.kitti_transform_test
        self.ds = torchvision.datasets.Kitti2015Stereo(root=root, split='train', transforms=self.transforms)
        
        # here we split 160 + 20 + 20, 160 train | 20 val | 20 test
        assert split in ('train', 'val', 'test')
        if split == 'train':
            self.ds._images = self.ds._images[:160*2]
        elif split == 'val':
            self.ds._images = self.ds._images[160*2:180*2]
        elif split == 'test':
            self.ds._images = self.ds._images[180*2:]
        
        self.ds._has_built_in_disparity_mask = False

    def __len__(self):
        return len(self.ds) // 2
    
    def __getitem__(self, index):
        return self.ds.__getitem__(index)

    def kitti_transform_train(self, imgs, dmap, valid_masks):
        
        img1 = imgs[0]
        img2 = imgs[1]

        dmap1 = dmap[0]
        dmap2 = dmap[1]
        
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        assert img1.shape == img2.shape
        oh, ow, oc = img1.shape
        
        # Collecting queries using FAST Feature detector
        # ...
        
        maxX, maxY = dmap1.shape[1:]
        fast = cv2.FastFeatureDetector_create()
        
        dmapt = dmap1.squeeze(0) > 0
        dmapt = dmapt.astype(np.uint8)
        
        imgt = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        
        kp = fast.detect(imgt, dmapt)
        
      
        #pix = cv2.drawKeypoints(img1, kp, None, color=(0,255,0))
        kp = [[int(p.pt[0]), int(p.pt[1])] for p in kp if p.pt[1] < maxX and p.pt[0] < maxY]
        # How to turn kp to queries (what shape ? np.array or dict or ?)
   
        targets = [ [int(p[0]-dmap1[0,p[1],p[0]]), p[1]] for p in kp]
        
        kp = np.array(kp).astype(np.float32)
        targets = np.array(targets).astype(np.float32)

        assert kp.dtype is np.dtype('float32')
        assert targets.dtype is np.dtype('float32')
        
        # CORRS TO 0-1 interval
        kp[:, 0] /= 2*ow
        kp[:, 1] /= oh

        #target masks
        tgt_mask = targets[:, 0] > 0

        targets[:, 0] /= 2*ow
        targets[:, 0] = targets[:, 0] + 0.5
        targets[:, 1] /= oh
        
        kp = kp[tgt_mask]
        targets = targets[tgt_mask]

        # COMBINE CORRS
        kp_shape = kp.shape
        t_shape = targets.shape
        
        kp = kp.reshape(1, kp_shape[0], kp_shape[1])
        targets = targets.reshape(1, t_shape[0], t_shape[1])
    
        corrs = np.concatenate((kp,targets), axis=0)
        mask = np.random.choice(corrs.shape[1], self.num_kp)
        corrs = corrs[:, mask, :]

        # BLUR
        ksize = (5,5)
        img1 = cv2.blur(img1, ksize)
        img2 = cv2.blur(img2, ksize)

        # RESIZING
        new_size = (self.img_size, self.img_size)
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_CUBIC)
    
        imgR = two_images_side_by_side(img1, img2)

        # TO TENSOR

        imgR = TF.to_tensor(imgR)
        #imgR = torch.tensor(imgR)
        #imgR = imgR.reshape(3,256,512)
        
        imgs = (imgR, img2)

        #dmap = (dmap1, dmap2)
    
        dmap = (corrs,dmap2)

        return imgs, dmap, valid_masks

    def kitti_transform_test(self, imgs, dmap, valid_masks):
        
        img1 = imgs[0]
        img2 = imgs[1]
        
        img1 = np.array(img1)
        img2 = np.array(img2)
        oh, ow, oc = img1.shape
        
        fast = cv2.FastFeatureDetector_create()
        
        imgt = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        
        kp = fast.detect(imgt, None)
        #pix = cv2.drawKeypoints(img1, kp, None, color=(255,0,0))
        
        pts = 20
        step = int(len(kp) / pts)
        kp = kp[::step]
        
        kp = [[int(p.pt[0]), int(p.pt[1])] for p in kp]
        kp = np.array(kp).astype(np.float32)
        kp[:, 0] /= 2*ow
        kp[:, 1] /= oh
        
        ksize = (5,5)
        img1 = cv2.blur(img1, ksize)
        img2 = cv2.blur(img2, ksize)
        
        new_size = (self.img_size, self.img_size)
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_CUBIC)
        
        imgR = two_images_side_by_side(img1, img2)
        
        imgR = TF.to_tensor(imgR)

        imgs = (imgR, img2)
        
        dmap = (kp, 0)
        
        return imgs, dmap, valid_masks

def test():

    ds = KittiDataset(root='../dataset/')
    i0 = ds[0]
    return
 
if __name__ == '__main__':
    test()
