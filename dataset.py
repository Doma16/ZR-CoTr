from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as t
import torch

class KittiDataset(Dataset):
    def __init__(self, root, split='train', transform = None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.ds = torchvision.datasets.Kitti2015Stereo(root=root, split=split, transforms=transform)
        print('CHECK LENGTH IN __len__, gives 400 should give 200', f'{self.ds.__len__}')

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds.__getitem__(index)

def test():

    ds = KittiDataset(root='../dataset/')

if __name__ == '__main__':
    test()