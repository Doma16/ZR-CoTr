import torch
import torchvision

class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        ## type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
    
def max_by_axis(l):
    maxs = l[0]
    for sub_l in l[1:]:
        for idx, item in enumerate(sub_l):
            maxs[idx] = max(maxs[idx], item)
    return maxs

def nested_tensor_from_tensor_list(tensor_list):
    
    if tensor_list[0].ndim == 3:
        max_size = max_by_axis([ list(img.shape) for img in tensor_list ])
        batch_shape = [len( tensor_list )] + max_size
        b,c,h,w = batch_shape
        dtype= tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b,h,w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError(f'nested_tensor_from_tensor_list func not supported {tensor_list[0].ndim}')
        
    return NestedTensor(tensor, mask)