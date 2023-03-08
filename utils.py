import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

IMAGE_SIZE = 256

def kitti_transform(self,img1, img2, dmap):
    
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))

    img1 = resize(img1)
    img2 = resize(img2)
    dmap = resize(dmap)

    # more transforms ?
    
    img1 = TF.to_tensor(img1)
    img2 = TF.to_tensor(img2)
    dmap = TF.to_tensor(dmap)



    return img1, img2, dmap