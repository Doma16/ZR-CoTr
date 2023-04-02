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
        
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        assert img1.shape == img2.shape
        oh, ow, oc = img1.shape
        
        #quick test.py fix, need better fix
        if dmap1 is None:
            ksize = (5,5)
            img1 = cv2.blur(img1, ksize)
            img2 = cv2.blur(img2, ksize)

            # RESIZING
            new_size = (self.img_size, self.img_size)
            img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_CUBIC)
        
            imgR = two_images_side_by_side(img1, img2)

            imgR = TF.to_tensor(imgR)
             
            imgs = (imgR, img2)
            
            dmap = (0,0)
            return imgs, dmap, valid_masks
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
        kp[:, 0] /= ow
        kp[:, 1] /= oh

        targets[:, 0] /= ow
        targets[:, 1] /= oh

        # COMBINE CORRS
        kp_shape = kp.shape
        t_shape = targets.shape
        
        kp = kp.reshape(1, kp_shape[0], kp_shape[1])
        targets = targets.reshape(1, t_shape[0], t_shape[1])
    
        corrs = np.concatenate((kp,targets), axis=0)
        
        # BLUR
        ksize = (5,5)
        img1 = cv2.blur(img1, ksize)
        img2 = cv2.blur(img2, ksize)

        # RESIZING
        new_size = (self.img_size, self.img_size)
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_CUBIC)
    
        imgR = two_images_side_by_side(img1, img2)
        '''
        resize = t.Resize(size=(self.img_size, self.img_size))
        
        img1 = resize(img1)
        img2 = resize(img2)

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
        '''
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        '''
        imgR = TF.to_tensor(imgR)
        #imgR = torch.tensor(imgR)
        #imgR = imgR.reshape(3,256,512)
        
        imgs = (imgR, img2)

        #dmap = (dmap1, dmap2)
    
        dmap = (corrs,dmap2)

        return imgs, dmap, valid_masks

def test():

    ds = KittiDataset(root='../dataset/')
    i0 = ds[0]
    return
 
if __name__ == '__main__':
    test()
