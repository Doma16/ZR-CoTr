from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as tform
import torchvision.transforms.functional as TF
import torch

import random
import numpy as np
from PIL import Image

import cv2
import matplotlib.pyplot as plt
from utils import two_images_side_by_side, two_images_vertical

class KittiDataset(Dataset):
    def __init__(self, root, split='train', transforms = None, img_size=256, num_kp=100):
        super().__init__()
        self.img_size = img_size
        self.root = root
        self.split = split
        self.num_kp = num_kp
        self.transforms = self.kitti_tile # if split == 'train' else self.kitti_transform_test
        self.transforms = self.kitti_zoom
        self.transforms = self.get_transform(transforms)
        self.ds = torchvision.datasets.Kitti2015Stereo(root=root, split='train', transforms=self.transforms)


        self.max = 0
        self.min = 100
        self.averages = []
        self.r_max = 0
        self.r_min = np.inf
        self.r_averages = []
        # here we split 160 + 40, 160 train | 40 val
        assert split in ('train', 'val')
        self.ds._images = self.ds._images[::2]
        if split == 'train':
            self.ds._images = self.ds._images[:160]
            self.ds._disparities = self.ds._disparities[:160]
        elif split == 'val':
            self.ds._images = self.ds._images[160:]
            self.ds._disparities = self.ds._disparities[160:]
        
        self.ds._has_built_in_disparity_mask = False

    def get_transform(self, tfs):
        if tfs == 'original':
            return self.kitti_original
        elif tfs == 'tile':
            return self.kitti_tile
        elif tfs == 'zoom':
            return self.kitti_zoom
        elif tfs == 'random_patch':
            return self.kitti_random_patch
        elif tfs == 'patch':
            return self.kitti_patch
        elif tfs == 'rect_patch':
            return self.kitti_rect_patch
        elif tfs == 's_patch':
            return self.kitti_s_patch
        else:
            return self.kitti_original


    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds.__getitem__(index)

    def kitti_transform_train(self, imgs, dmap, valid_masks):
        
        img1 = imgs[0]
        img2 = imgs[1]

        dmap1 = dmap[0]
        dmap2 = dmap[1]
        
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        ''' prikaz rektifikacije
        canvas = two_images_side_by_side(img1, img2)
        plt.imshow(canvas)

        space = np.linspace(0,canvas.shape[0], 9)[1:-1]

        for line in space:
            plt.axhline(y=line, color='c', linestyle='-', linewidth=1)

        plt.axis('off')
        plt.show()
        '''
        
        ''' prikaz mape dispariteta
        plt.imshow(dmap1[0], cmap='gray')
        plt.axis('off')
        plt.show()
        '''

        indicies = np.argwhere(dmap1[0] > 0)
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
        targets = [ [round(p[0]-dmap1[0,p[1],p[0]]), p[1]] for p in kp]
        
        #used for debugging
        '''
        breakpoint()
        kp = np.array(kp)
        targets = np.array(targets)

        mask = np.random.choice(kp.shape[0], 100)
        kp = kp[mask, :]
        targets = targets[mask, :]

        fig, axes = plt.subplots(1,2)
        axes[0].imshow(img1)
        axes[0].scatter(kp[:,0], kp[:,1], c='red', marker='x')

        axes[1].imshow(img2)
        axes[1].scatter(targets[:,0], targets[:, 1], c='blue', marker='x')

        plt.show()
        breakpoint()
        '''

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

        # try some zoom in?
        if False:
            x_start = 0.5
            x_end = 1.0
            x_min = np.min(corrs[1,:,0])
            x_max = np.max(corrs[1,:,0])

            y_min = np.min(corrs[1,:,1])
            y_max = np.max(corrs[1,:,1])

            h,w,c = img2.shape
            new_x_start = round((x_min-0.5)*2*w)
            new_x_end = round((x_max-0.5)*2*w)
            new_y_start = round((y_min)*h)
            new_y_end = round((y_max)*h)
            img2 = img2[new_y_start:new_y_end,new_x_start:new_x_end,:]

            corrs[1,:,0] = (corrs[1,:,0] - x_min) / (x_max-x_min) / 2 + 0.5
            corrs[1,:,1] = (corrs[1,:,1] - y_min) / (y_max-y_min)


        # RESIZING
        new_size = (self.img_size, self.img_size)
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_LINEAR)
    
        imgR = two_images_side_by_side(img1, img2)

        # TO TENSOR

        imgR = TF.to_tensor(imgR)
        #imgR = torch.tensor(imgR)
        #imgR = imgR.reshape(3,256,512)
        imgReal = two_images_vertical(np.array(imgs[0]), np.array(imgs[1]))
        imgs = (imgR, 1)

        #dmap = (dmap1, dmap2)
    
        dmap = (corrs,1)

        return imgs, dmap, valid_masks
    
    def kitti_tile(self, imgs, dmap, valid_masks):        
        img1 = imgs[0]
        img2 = imgs[1]

        img1 = np.array(img1)
        img2 = np.array(img2)

        oh, ow, oc = img1.shape
        dmap1 = dmap[0]
        dmap2 = dmap[1]

        indicies = np.argwhere(dmap1[0] > 0)[2*ow:-2*ow]
        temp = np.copy(dmap1[0])
        temp = temp[temp > 0][2*ow:-2*ow]

        assert img1.shape == img2.shape
        assert img1.shape[:2] == dmap1.shape[1:]
        maxX, maxY = dmap1.shape[1:]

        miny = indicies[:,0].min()
        maxy = indicies[:,0].max()

        minx = indicies[:,1].min()
        maxx = indicies[:,1].max()

        img1 = img1[miny:maxy,:,:]
        img2 = img2[miny:maxy,:,:]

        indicies2 = np.copy(indicies).astype(np.float32)
        indicies2[:,1] = indicies2[:,1] - temp 

        tgt_mask1 = indicies2[:,1] > 0
        tgt_mask2 = indicies2[:,1] < ow
        tgt_mask = np.logical_and(tgt_mask1, tgt_mask2)

        indicies2 = indicies2[tgt_mask]
        indicies = indicies[tgt_mask]

        diff = maxy - miny
        indicies = indicies.astype(np.float32)
        indicies[:,0] = indicies[:,0] - miny

        indicies2 = indicies2.astype(np.float32)
        indicies2[:,0] = indicies2[:,0] - miny

        indicies = np.round(indicies)
        indicies2 = np.round(indicies2)


        indicies[:, [0,1]] = indicies[:, [1,0]]
        indicies2[:, [0,1]] = indicies2[:, [1,0]]
        
        indicies[:, 0] /= 2*ow
        indicies[:, 1] /= diff
        indicies2[:, 0] /= 2*ow
        indicies2[:, 0] = indicies2[:, 0] + 0.5
        indicies2[:, 1] /= diff

        indicies = indicies.reshape(1, indicies.shape[0], indicies.shape[1])
        indicies2 = indicies2.reshape(1, indicies2.shape[0], indicies2.shape[1])
     
        corrs = np.concatenate((indicies, indicies2), axis=0)
        mask = np.random.choice(corrs.shape[1], self.num_kp)
        curr_val_resize = (256*2*(corrs[0, : , 0] - (corrs[1, : , 0]-0.5))).max()

        self.r_averages.append(curr_val_resize)
        if curr_val_resize > self.r_max:
            self.r_max = curr_val_resize
        if curr_val_resize < self.r_min:
            self.r_min = curr_val_resize

        print()
        print('img1_r: ',img1.shape)
        print('img2_r: ',img2.shape)
        print('max_disp_r: ', self.r_max)
        print('min_disp_r: ', self.r_min)
        print('avg_disp_r: ', sum(self.r_averages)/len(self.r_averages))
        
        corrs = corrs[:, mask, :]

        new_size = (self.img_size, self.img_size)
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_LINEAR)
        
        '''
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(img1)
        axes[0].scatter(np.round(corrs[0,:,0]*2*256), np.round(corrs[0,:,1]*256), c='red', marker='x')
        axes[0].axis('off')

        axes[1].imshow(img2)
        axes[1].scatter(np.round((corrs[1,:,0]-0.5)*2*256), np.round(corrs[1,:,1]*256), c='blue', marker='x')
        axes[1].axis('off')

        plt.show()
        breakpoint()
        '''
        imgR = two_images_side_by_side(img1, img2)

        imgR = TF.to_tensor(imgR)
        imgs = (imgR, 1)
        dmap = (corrs, 1)

        return imgs, dmap, valid_masks

    def kitti_zoom(self, imgs, dmap, valid_masks):        
        img1 = imgs[0]
        img2 = imgs[1]

        img1 = np.array(img1)
        img2 = np.array(img2)

        oh, ow, oc = img1.shape
        dmap1 = dmap[0]
        dmap2 = dmap[1]

        max_d = dmap1.max()

        indicies = np.argwhere(dmap1[0] > 0)[2*ow:-2*ow]
        temp = np.copy(dmap1[0])
        temp = temp[temp > 0][2*ow:-2*ow]

        assert img1.shape == img2.shape
        assert img1.shape[:2] == dmap1.shape[1:]
        maxX, maxY = dmap1.shape[1:]

        miny = indicies[:,0].min()
        maxy = indicies[:,0].max()

        minx = indicies[:,1].min()
        maxx = indicies[:,1].max()

        img1 = img1[miny:maxy,:,:]
        img2 = img2[miny:maxy,:,:]

        indicies2 = np.copy(indicies).astype(np.float32)
        indicies2[:,1] = indicies2[:,1] - temp 

        tgt_mask1 = indicies2[:,1] > 0
        tgt_mask2 = indicies2[:,1] < ow
        tgt_mask = np.logical_and(tgt_mask1, tgt_mask2)

        indicies2 = indicies2[tgt_mask]
        indicies = indicies[tgt_mask]

        diff = maxy - miny
        indicies = indicies.astype(np.float32)
        indicies[:,0] = indicies[:,0] - miny

        indicies2 = indicies2.astype(np.float32)
        indicies2[:,0] = indicies2[:,0] - miny

        indicies = np.round(indicies)
        indicies2 = np.round(indicies2)


        indicies[:, [0,1]] = indicies[:, [1,0]]
        indicies2[:, [0,1]] = indicies2[:, [1,0]]

        #zoom in x coords
        start_x = random.randint(0,ow-self.img_size)
        img1 = img1[:,start_x:start_x+self.img_size,:]


        mask_shift = np.logical_and(indicies[:,0] > start_x, indicies[:,0] < start_x+self.img_size)

        indicies = indicies[mask_shift]
        indicies2 = indicies2[mask_shift]

        indicies[:,0] = indicies[:, 0] - start_x
        #
        start_x2 = int(max(0, start_x-max_d))
        end_x2 = int(min(start_x+self.img_size+max_d, img2.shape[1]))

        img2 = img2[:, start_x2:end_x2, :]

        mask_shift2 = np.logical_and(indicies2[:, 0] > start_x2, indicies2[:,0] < start_x+end_x2)

        indicies = indicies[mask_shift2]
        indicies2 = indicies2[mask_shift2]

        indicies2[:,0] = indicies2[:,0] - start_x2

        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        #zoom y
        start_y = random.randint(0,20)
        start_y = min(start_y, h1-start_y)

        img1 = img1[start_y:h1-start_y,:,:]

        mask_shift_y = np.logical_and(indicies[:, 1] > start_y, indicies[:,1] < h1-start_y)

        indicies = indicies[mask_shift_y]
        indicies2 = indicies2[mask_shift_y]

        indicies[:, 1] = indicies[:, 1] - start_y
        
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        indicies[:, 0] /= 2*w1
        indicies[:, 1] /= h1
        indicies2[:, 0] /= 2*w2
        indicies2[:, 0] = indicies2[:, 0] + 0.5
        indicies2[:, 1] /= h2

        indicies = indicies.reshape(1, indicies.shape[0], indicies.shape[1])
        indicies2 = indicies2.reshape(1, indicies2.shape[0], indicies2.shape[1])
     
        corrs = np.concatenate((indicies, indicies2), axis=0)
        if corrs.shape[1] == 0:
            return self.kitti_tile(imgs, dmap, valid_masks)
        mask = np.random.choice(corrs.shape[1], self.num_kp)
        corrs = corrs[:, mask, :]

        new_size = (self.img_size, self.img_size)
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_LINEAR)
        
        '''
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(img1)
        axes[0].scatter(np.round(corrs[0,:,0]*2*256), np.round(corrs[0,:,1]*256), c='red', marker='x')

        axes[1].imshow(img2)
        axes[1].scatter(np.round((corrs[1,:,0]-0.5)*2*256), np.round(corrs[1,:,1]*256), c='blue', marker='x')

        plt.show()
        breakpoint()
        '''
        
        imgR = two_images_side_by_side(img1, img2)

        imgR = TF.to_tensor(imgR)
        imgs = (imgR, 1)
        dmap = (corrs, 1)
        return imgs, dmap, valid_masks
    
    def kitti_random_patch(self, imgs, dmap, valid_masks):
        img1 = imgs[0]
        img2 = imgs[1]

        img1 = np.array(img1)
        img2 = np.array(img2)

        dmap1 = dmap[0]
        dmap2 = dmap[1]

        indicies = np.argwhere(dmap1[0] > 0)
        temp = np.copy(dmap1[0])
        temp = temp[temp > 0]
        
        indicies2 = np.copy(indicies).astype(np.float32)
        indicies2[:, 1] = indicies2[:, 1] - temp

        h2,w2,_ = img2.shape
        
        tgt_mask1 = indicies2[:, 1] > 0
        tgt_mask2 = indicies2[:, 1] < w2
        tgt_mask = np.logical_and(tgt_mask1, tgt_mask2)

        indicies2 = indicies2[tgt_mask]
        indicies = indicies[tgt_mask]

        temp = temp[tgt_mask]

        indicies = indicies.astype(np.float32)
        indicies2 = indicies2.astype(np.float32)
        
        indicies = np.round(indicies)
        indicies2 = np.round(indicies2)

        indicies[:, [0,1]] = indicies[:, [1,0]]
        indicies2[:, [0,1]] = indicies2[:, [1,0]] 
    
        mask = np.random.choice(indicies.shape[0], self.num_kp)
        indicies = indicies[mask]
        indicies2 = indicies2[mask]
        temp = temp[indicies]

        breakpoint()
        h1,w1,_ = img1.shape

        w_min = indicies[:, 0].min()
        w_max = indicies[:, 0].max()

        h_min = indicies[:, 1].min()
        h_max = indicies[:, 1].max()
    
        # crop coords for img1
        ws = random.randint(0, w_min)
        we = random.randint(w_max, w1)
        hs = random.randint(0, h_min)
        he = random.randint(h_max, h1)

        #crop coords for img2

        breakpoint()
        return imgs, dmap, valid_masks
    
    def kitti_patch(self, imgs, dmap, valid_masks):

        img1 = imgs[0]
        img2 = imgs[1]

        img1 = np.array(img1)
        img2 = np.array(img2)


        dmap1 = dmap[0]
        dmap2 = dmap[1]
        
        curr_val = dmap1.max()
        self.averages.append(curr_val)
        if self.max < curr_val:
            self.max = curr_val
        if self.min > curr_val:
            self.min = curr_val

        '''
        '''
        print()
        print(f'img1: {img1.shape}')
        print(f'img2: {img2.shape}')
        print('max_disp: ',self.max)
        print('min_disp: ',self.min)
        print('avg_disp: ',sum(self.averages)/len(self.averages))

        indicies = np.argwhere(dmap1[0] > 0)
        temp = np.copy(dmap1[0])
        disps = temp[temp > 0]

        indicies = indicies.astype(np.float32)
        indicies2 = np.copy(indicies).astype(np.float32)
        indicies2[:, 1] = indicies2[:, 1] - disps

        indicies[:, [0,1]] = indicies[:, [1,0]]
        indicies2[:, [0,1]] = indicies2[:, [1,0]] 

        #random crop coords img1
        h1,w1,_ = img1.shape
        
        x_c = random.randint(0, w1-self.img_size)
        y_c = random.randint(0, h1-self.img_size)

        img1 = img1[y_c:y_c+self.img_size, x_c:x_c+self.img_size, :]

        mask_x1 = indicies[:, 0] > x_c
        mask_x2 = indicies[:, 0] < x_c + self.img_size
        mask_x = np.logical_and(mask_x1, mask_x2)
        mask_y1 = indicies[:, 1] > y_c
        mask_y2 = indicies[:, 1] < y_c + self.img_size
        mask_y = np.logical_and(mask_y1, mask_y2)
        mask_xy = np.logical_and(mask_x, mask_y)


        indicies = indicies[mask_xy]
        indicies2 = indicies2[mask_xy]
        disps = disps[mask_xy]

        indicies[:, 0] = indicies[:, 0] - x_c
        indicies[:, 1] = indicies[:, 1] - y_c

        mask_x_ = indicies2[:, 0] > 0
        indicies2 = indicies2[mask_x_]
        indicies = indicies[mask_x_]
        disps = disps[mask_x_]

        #check if indicies contain as much kp_s self.num_kp
        
        h2,w2,_ = img2.shape

        if indicies2.shape[0] == 0:
            return self.kitti_patch(imgs, dmap, valid_masks)

        x_s = indicies2[:, 0].min()
        x_e = indicies2[:, 0].max()
        
        y_s = indicies2[:, 1].min()
        y_e = indicies2[:, 1].max()

        diff_y = self.img_size - (y_e - y_s)

        tt = max(0,int(y_s - diff_y))
        tt2 = min(h2, y_e + diff_y) - self.img_size
        y2_c = random.randint(tt, tt2)
        
        diff_x = self.img_size - (x_e - x_s)
        if diff_x < 0:
            x2_c = random.randint(int(x_s), int(x_e - self.img_size))
        else:
            x2_c = random.randint(max(0,int(x_s - diff_x)), min(w2, int(x_e + diff_x)) - self.img_size)
        # cropping

        if x2_c < 0:
            x2_c += -x2_c

        img2 = img2[y2_c:y2_c+self.img_size,x2_c:x2_c+self.img_size,:]

        mask2_x1 = indicies2[:, 0] > x2_c
        mask2_x2 = indicies2[:, 0] < x2_c + self.img_size
        mask2_x = np.logical_and(mask2_x1, mask2_x2)
        mask2_y1 = indicies2[:, 1] > y2_c
        mask2_y2 = indicies2[:, 1] < y2_c + self.img_size
        mask2_y = np.logical_and(mask2_y1, mask2_y2)
        mask2_xy = np.logical_and(mask2_x, mask2_y)

        indicies = indicies[mask2_xy]
        indicies2 = indicies2[mask2_xy]
        disps = disps[mask2_xy]

        indicies2[:, 0] = indicies2[:, 0] - x2_c
        indicies2[:, 1] = indicies2[:, 1] - y2_c

        if indicies.shape[0] < self.num_kp:
            return self.kitti_patch(imgs, dmap, valid_masks)
        #assert indicies.shape[0] > self.num_kp

        indicies[:, 0] /= 2*self.img_size
        indicies[:, 1] /= self.img_size
        indicies2[:, 0] /= 2*self.img_size
        indicies2[:, 0] = indicies2[:, 0] + 0.5
        indicies2[:, 1] /= self.img_size

        indicies = indicies.reshape(1, indicies.shape[0], indicies.shape[1])
        indicies2 = indicies2.reshape(1, indicies2.shape[0], indicies2.shape[1])

        corrs = np.concatenate((indicies, indicies2), axis=0)
        mask = np.random.choice(corrs.shape[1], self.num_kp)
        corrs = corrs[:,mask,:]

        '''
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(img1)
        axes[0].scatter(np.round(corrs[0,:,0]*2*256), np.round(corrs[0,:,1]*256), c='red', marker='x')
        axes[0].axis('off')

        axes[1].imshow(img2)
        axes[1].scatter(np.round((corrs[1,:,0]-0.5)*2*256), np.round(corrs[1,:,1]*256), c='blue', marker='x')
        axes[1].axis('off')

        plt.show()
        breakpoint()
        '''

        curr_val_resize = (256*2*(corrs[0, : , 0] - (corrs[1, : , 0]-0.5))).max()

        self.r_averages.append(curr_val_resize)
        if curr_val_resize > self.r_max:
            self.r_max = curr_val_resize
        if curr_val_resize < self.r_min:
            self.r_min = curr_val_resize

        print()
        print('img1_r: ',img1.shape)
        print('img2_r: ',img2.shape)
        print('max_disp_r: ', self.r_max)
        print('min_disp_r: ', self.r_min)
        print('avg_disp_r: ', sum(self.r_averages)/len(self.r_averages))
        
        imgR = two_images_side_by_side(img1, img2)
        imgR = TF.to_tensor(imgR)
        imgs = (imgR, 1)
        dmap = (corrs, 1)

        return imgs, dmap, valid_masks

    def kitti_rect_patch(self, imgs, dmap, valid_masks):

        img1 = imgs[0]
        img2 = imgs[1]

        img1 = np.array(img1)
        img2 = np.array(img2)

        dmap1 = dmap[0]
        dmap2 = dmap[1]

        indicies = np.argwhere(dmap1[0] > 0)
        temp = np.copy(dmap1[0])
        disps = temp[temp > 0]

        indicies = indicies.astype(np.float32)
        indicies2 = np.copy(indicies).astype(np.float32)
        indicies2[:, 1] = indicies2[:, 1] - disps

        indicies[:, [0,1]] = indicies[:, [1,0]]
        indicies2[:, [0,1]] = indicies2[:, [1,0]] 

        #random crop coords img1
        h1,w1,_ = img1.shape
        
        x_c = random.randint(0, w1-self.img_size)
        y_c = random.randint(0, h1-self.img_size)

        img1 = img1[y_c:y_c+self.img_size, x_c:x_c+self.img_size, :]

        mask_x1 = indicies[:, 0] > x_c
        mask_x2 = indicies[:, 0] < x_c + self.img_size
        mask_x = np.logical_and(mask_x1, mask_x2)
        mask_y1 = indicies[:, 1] > y_c
        mask_y2 = indicies[:, 1] < y_c + self.img_size
        mask_y = np.logical_and(mask_y1, mask_y2)
        mask_xy = np.logical_and(mask_x, mask_y)


        indicies = indicies[mask_xy]
        indicies2 = indicies2[mask_xy]
        disps = disps[mask_xy]

        indicies[:, 0] = indicies[:, 0] - x_c
        indicies[:, 1] = indicies[:, 1] - y_c

        mask_x_ = indicies2[:, 0] > 0
        indicies2 = indicies2[mask_x_]
        indicies = indicies[mask_x_]
        disps = disps[mask_x_]

        #check if indicies contain as much kp_s self.num_kp
        
        h2,w2,_ = img2.shape

        if indicies2.shape[0] == 0:
            return self.kitti_rect_patch(imgs, dmap, valid_masks)

        x_s = indicies2[:, 0].min()
        x_e = indicies2[:, 0].max()
        
        y_s = indicies2[:, 1].min()
        y_e = indicies2[:, 1].max()

        diff_y = self.img_size - (y_e - y_s)

        tt = max(0,int(y_s - diff_y))
        tt2 = min(h2, y_e + diff_y) - self.img_size
        y2_c = random.randint(tt, tt2)
        y2_c = y_c
        
        diff_x = self.img_size - (x_e - x_s)
        if diff_x < 0:
            x2_c = random.randint(int(x_s), int(x_e - self.img_size))
        else:
            x2_c = random.randint(max(0,int(x_s - diff_x)), min(w2, int(x_e + diff_x)) - self.img_size)
        # cropping

        if x2_c < 0:
            x2_c += -x2_c

        img2 = img2[y2_c:y2_c+self.img_size,x2_c:x2_c+self.img_size,:]

        mask2_x1 = indicies2[:, 0] > x2_c
        mask2_x2 = indicies2[:, 0] < x2_c + self.img_size
        mask2_x = np.logical_and(mask2_x1, mask2_x2)
        mask2_y1 = indicies2[:, 1] > y2_c
        mask2_y2 = indicies2[:, 1] < y2_c + self.img_size
        mask2_y = np.logical_and(mask2_y1, mask2_y2)
        mask2_xy = np.logical_and(mask2_x, mask2_y)

        indicies = indicies[mask2_xy]
        indicies2 = indicies2[mask2_xy]
        disps = disps[mask2_xy]

        indicies2[:, 0] = indicies2[:, 0] - x2_c
        indicies2[:, 1] = indicies2[:, 1] - y2_c

        if indicies.shape[0] < self.num_kp:
            return self.kitti_rect_patch(imgs, dmap, valid_masks)
        #assert indicies.shape[0] > self.num_kp

        indicies[:, 0] /= 2*self.img_size
        indicies[:, 1] /= self.img_size
        indicies2[:, 0] /= 2*self.img_size
        indicies2[:, 0] = indicies2[:, 0] + 0.5
        indicies2[:, 1] /= self.img_size

        indicies = indicies.reshape(1, indicies.shape[0], indicies.shape[1])
        indicies2 = indicies2.reshape(1, indicies2.shape[0], indicies2.shape[1])

        corrs = np.concatenate((indicies, indicies2), axis=0)
        mask = np.random.choice(corrs.shape[1], self.num_kp)
        corrs = corrs[:,mask,:]

        '''
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(img1)
        axes[0].scatter(np.round(corrs[0,:,0]*2*256), np.round(corrs[0,:,1]*256), c='red', marker='x')

        axes[1].imshow(img2)
        axes[1].scatter(np.round((corrs[1,:,0]-0.5)*2*256), np.round(corrs[1,:,1]*256), c='blue', marker='x')

        plt.show()
        breakpoint()
        '''
        
        imgR = two_images_side_by_side(img1, img2)
        imgR = TF.to_tensor(imgR)
        imgs = (imgR, 1)
        dmap = (corrs, 1)

        return imgs, dmap, valid_masks
    
    def kitti_s_patch(self, imgs, dmap, valid_masks):

        edge_threshold = 8

        img1 = imgs[0]
        img2 = imgs[1]

        img1 = np.array(img1)
        img2 = np.array(img2)

        dmap1 = dmap[0]
        dmap2 = dmap[1]

        d_value = 1

        indicies = np.argwhere(dmap1[0] > d_value)
        temp = np.copy(dmap1[0])
        disps = temp[temp > d_value]

        indicies = indicies.astype(np.float32)
        indicies2 = np.copy(indicies).astype(np.float32)
        indicies2[:, 1] = indicies2[:, 1] - disps

        indicies[:, [0,1]] = indicies[:, [1,0]]
        indicies2[:, [0,1]] = indicies2[:, [1,0]] 

        #random crop coords img1
        h1,w1,_ = img1.shape
        
        x_c = random.randint(0, w1-self.img_size)
        y_c = random.randint(0, h1-self.img_size)

        img1 = img1[y_c:y_c+self.img_size, x_c:x_c+self.img_size, :]

        mask_x1 = indicies[:, 0] > x_c + edge_threshold
        mask_x2 = indicies[:, 0] < x_c + self.img_size - edge_threshold 
        mask_x = np.logical_and(mask_x1, mask_x2)
        mask_y1 = indicies[:, 1] > y_c + edge_threshold
        mask_y2 = indicies[:, 1] < y_c + self.img_size - edge_threshold
        mask_y = np.logical_and(mask_y1, mask_y2)
        mask_xy = np.logical_and(mask_x, mask_y)


        indicies = indicies[mask_xy]
        indicies2 = indicies2[mask_xy]
        disps = disps[mask_xy]

        indicies[:, 0] = indicies[:, 0] - x_c
        indicies[:, 1] = indicies[:, 1] - y_c

        mask_x_ = indicies2[:, 0] > 0
        indicies2 = indicies2[mask_x_]
        indicies = indicies[mask_x_]
        disps = disps[mask_x_]

        #check if indicies contain as much kp_s self.num_kp
        
        h2,w2,_ = img2.shape

        if indicies2.shape[0] == 0:
            return self.kitti_s_patch(imgs, dmap, valid_masks)

        x_s = indicies2[:, 0].min()
        x_e = indicies2[:, 0].max()
        
        y_s = indicies2[:, 1].min()
        y_e = indicies2[:, 1].max()

        diff_y = self.img_size - (y_e - y_s)

        tt = max(0,int(y_s - diff_y))
        tt2 = min(h2, y_e + diff_y) - self.img_size
        y2_c = random.randint(tt, tt2)
        y2_c = y_c
        
        diff_x = self.img_size - (x_e - x_s)
        if diff_x < 0:
            x2_c = random.randint(int(x_s), int(x_e - self.img_size))
        else:
            x2_c = random.randint(max(0,int(x_s - diff_x)), min(w2, int(x_e + diff_x)) - self.img_size)
        # cropping

        if x2_c < 0:
            x2_c += -x2_c

        r_c = random.choice([128, 96, 64, 32, 0])
        x2_c = max(0, round(x_s) - r_c)

        img2 = img2[y2_c:y2_c+self.img_size,x2_c:x2_c+self.img_size,:]

        mask2_x1 = indicies2[:, 0] > x2_c + edge_threshold
        mask2_x2 = indicies2[:, 0] < x2_c + self.img_size - edge_threshold
        mask2_x = np.logical_and(mask2_x1, mask2_x2)
        mask2_y1 = indicies2[:, 1] > y2_c + edge_threshold
        mask2_y2 = indicies2[:, 1] < y2_c + self.img_size - edge_threshold
        mask2_y = np.logical_and(mask2_y1, mask2_y2)
        mask2_xy = np.logical_and(mask2_x, mask2_y)

        indicies = indicies[mask2_xy]
        indicies2 = indicies2[mask2_xy]
        disps = disps[mask2_xy]

        indicies2[:, 0] = indicies2[:, 0] - x2_c
        indicies2[:, 1] = indicies2[:, 1] - y2_c

        if indicies.shape[0] < self.num_kp:
            return self.kitti_s_patch(imgs, dmap, valid_masks)
        #assert indicies.shape[0] > self.num_kp

        indicies[:, 0] /= 2*self.img_size
        indicies[:, 1] /= self.img_size
        indicies2[:, 0] /= 2*self.img_size
        indicies2[:, 0] = indicies2[:, 0] + 0.5
        indicies2[:, 1] /= self.img_size

        indicies = indicies.reshape(1, indicies.shape[0], indicies.shape[1])
        indicies2 = indicies2.reshape(1, indicies2.shape[0], indicies2.shape[1])

        corrs = np.concatenate((indicies, indicies2), axis=0)
        mask = np.random.choice(corrs.shape[1], self.num_kp)
        corrs = corrs[:,mask,:]

        '''
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(img1)
        axes[0].scatter(np.round(corrs[0,:,0]*2*256), np.round(corrs[0,:,1]*256), c='red', marker='x')

        axes[1].imshow(img2)
        axes[1].scatter(np.round((corrs[1,:,0]-0.5)*2*256), np.round(corrs[1,:,1]*256), c='blue', marker='x')

        plt.show()
        breakpoint()
        '''

        imgR = two_images_side_by_side(img1, img2)
        imgR = TF.to_tensor(imgR)
        #imgR = TF.normalize(TF.to_tensor(imgR), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).float()
        imgs = (imgR, 1)
        dmap = (corrs, 1)

        return imgs, dmap, valid_masks

    def kitti_original(self, imgs, dmap, valid_masks):

        '''
        fig, axes = plt.subplots(2,1)
        axes[0].imshow(imgs[0])
        axes[1].imshow(dmap[0][0], cmap='gray')
        axes[0].axis('off')
        axes[1].axis('off')
        plt.show()
        breakpoint()
        '''
        img1 = imgs[0]
        img2 = imgs[1]
        
        img1 = np.array(img1)
        img2 = np.array(img2)

        dmap1 = dmap[0]
        dmap2 = dmap[1]

        indicies = np.argwhere(dmap1[0] > 0)
        
        # temp is disparity
        temp = np.copy(dmap1[0])
        temp = temp[temp > 0]

        indicies2 = np.copy(indicies).astype(np.float32)
        indicies2[:, 1] = indicies2[:, 1] - temp 

        h2,w2,_ = img2.shape

        tgt_mask1 = indicies2[:, 1] > 0
        tgt_mask2 = indicies2[:, 1] < w2
        tgt_mask = np.logical_and(tgt_mask1, tgt_mask2)

        indicies2 = indicies2[tgt_mask]
        indicies = indicies[tgt_mask]

        indicies = indicies.astype(np.float32)
        indicies2 = indicies2.astype(np.float32)

        indicies = np.round(indicies)
        indicies2 = np.round(indicies2)

        indicies[:, [0,1]] = indicies[:, [1,0]]
        indicies2[:, [0,1]] = indicies2[:, [1,0]]

        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        indicies[:, 0] /= 2*w1
        indicies[:, 1] /= h1

        indicies2[:, 0] /= 2*w2
        indicies2[:, 0] = indicies2[:, 0] + 0.5 
        indicies2[:, 1] /= h2

        indicies = indicies.reshape(1, indicies.shape[0], indicies.shape[1])
        indicies2 = indicies2.reshape(1, indicies2.shape[0], indicies2.shape[1])

        corrs = np.concatenate((indicies, indicies2), axis=0)
        mask = np.random.choice(corrs.shape[1], self.num_kp)
        corrs = corrs[:, mask, :]

        imgR = two_images_side_by_side(img1, img2)
        imgR = TF.to_tensor(imgR)

        imgs = (imgR, dmap[0][0])
        dmap = (corrs, 1)
    
        return imgs, dmap, valid_masks

class MiddleBury(Dataset):
    def __init__(self, root, split='train', transforms = None, img_size=256, num_kp=100, download=False):
        super().__init__()
        self.min = np.inf
        self.max = 0
        self.averages = []
        self.r_min = np.inf
        self.r_max = 0
        self.r_averages = []
        self.img_size = img_size
        self.root = root
        self.split = split
        self.num_kp = num_kp
        self.transforms = self.MiddleburyTransform
        self.transforms = self.get_transform(transforms)
        self.ds = torchvision.datasets.Middlebury2014Stereo(root=root, split=split, transforms=self.transforms, download=download)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds.__getitem__(index)
    
    def get_transform(self, tfs):
        if tfs == 'original':
            return self.MB_original
        else:
            return self.MiddleburyTransform
    
    def MB_original(self, imgs, dmap, valid_masks):
        img1 = imgs[0]
        img2 = imgs[1]

        '''
        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(imgs[0])
        axes[1].imshow(dmap[0][0], cmap='gray')
        axes[0].axis('off')
        axes[1].axis('off')
        
        plt.show()
        breakpoint()
        '''

        img1 = np.array(img1)
        img2 = np.array(img2)

        imgR = two_images_side_by_side(img1, img2)
        imgR = TF.to_tensor(imgR)
       
        imgs = (imgR, dmap[0][0])

        return imgs, dmap, valid_masks

    def MiddleburyTransform(self, imgs, dmap, valid_masks):
        img1 = imgs[0]
        img2 = imgs[1]

        img1 = np.array(img1)
        img2 = np.array(img2)

        oh, ow, oc = img1.shape

        dmap1 = dmap[0]
        dmap2 = dmap[1]

        curr_val = dmap1.max()
        if curr_val > self.max:
            self.max = curr_val
        if curr_val < self.min:
            self.min = curr_val
        self.averages.append(curr_val)

        print()
        print('img1: ',img1.shape)
        print('img2: ',img2.shape)
        print('max_disp: ', self.max)
        print('min_disp: ', self.min)
        print('avg_disp: ', sum(self.averages)/len(self.averages))

        vm1 = valid_masks[0]
        vm2 = valid_masks[1]

        new_mask = np.logical_and(dmap1, vm1)

        indicies = np.argwhere(dmap1[0] > 0.)
        temp = np.copy(dmap1[0])
        temp = temp[temp > 0.]

        indicies = indicies.astype(np.float32)
        indicies2 = np.copy(indicies).astype(np.float32)
        indicies2[:,1] = indicies2[:,1] - temp

        tgt_mask1 = indicies2[:, 1] > 0
        tgt_mask2 = indicies2[:, 1] < ow
        tgt_mask = np.logical_and(tgt_mask1, tgt_mask2)

        indicies2 = indicies2[tgt_mask]
        indicies = indicies[tgt_mask]

        indicies = np.round(indicies)
        indicies2 = np.round(indicies2)

        indicies[:, [0,1]] = indicies[:, [1,0]]
        indicies2[:, [0,1]] = indicies2[:, [1,0]]

        indicies[:, 0] /= 2*ow
        indicies[:, 1] /= oh
        indicies2[:, 0] /= 2*ow
        indicies2[:, 0] = indicies2[:, 0] + 0.5
        indicies2[:, 1] /= oh

        indicies = indicies.reshape(1, indicies.shape[0], indicies.shape[1])
        indicies2 = indicies2.reshape(1, indicies2.shape[0], indicies2.shape[1])
        
        corrs = np.concatenate((indicies, indicies2), axis=0)
        mask = np.random.choice(corrs.shape[1], self.num_kp)
        corrs = corrs[:, mask, :]

        new_size = (self.img_size, self.img_size)
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_LINEAR)

        '''
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(img1)
        axes[0].scatter(np.round(corrs[0,:,0]*2*256), np.round(corrs[0,:,1]*256), c='red', marker='x')

        axes[1].imshow(img2)
        axes[1].scatter(np.round((corrs[1,:,0]-0.5)*2*256), np.round(corrs[1,:,1]*256), c='blue', marker='x')

        plt.show()
        '''
        curr_val_resize = (256*2*(corrs[0, : , 0] - (corrs[1, : , 0]-0.5))).max()

        self.r_averages.append(curr_val_resize)
        if curr_val_resize > self.r_max:
            self.r_max = curr_val_resize
        if curr_val_resize < self.r_min:
            self.r_min = curr_val_resize

        print()
        print('img1_r: ',img1.shape)
        print('img2_r: ',img2.shape)
        print('max_disp_r: ', self.r_max)
        print('min_disp_r: ', self.r_min)
        print('avg_disp_r: ', sum(self.r_averages)/len(self.r_averages))

        imgR = two_images_side_by_side(img1, img2)
        imgR = TF.to_tensor(imgR)

        imgs = (imgR, 1)
        dmap = (corrs, 1)

        return imgs, dmap, valid_masks


        breakpoint()
        pass



def test():

    ds = KittiDataset(root='../dataset/')
    i0 = ds[0]
    return
 
if __name__ == '__main__':
    test()
