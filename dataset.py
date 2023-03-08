from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as t
import torch

print(torch.__version__)

class KittiDataset(Dataset):
    def __init__(self, root, split='train', transform = None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        ds = torchvision.datasets.Kitti2015Stereo(root=root, split=split, transforms=transform)

        breakpoint()


def test():

    ds = KittiDataset(root='../dataset/')

if __name__ == '__main__':
    test()