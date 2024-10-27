import math
from einops import rearrange
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from IPython.display import display, Image as IPythonImage
import io

class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])

def tensor_as_gif(images):
    if images.is_cuda:
        images = images.cpu()
    images_np = images.numpy()  # Shape: (16, 3, 192, 192)
    images_np = np.transpose(images_np, (0, 2, 3, 1))  # Shape: (16, 192, 192, 3)
    if images_np.max() <= 1.0:
        images_np = (images_np * 255).astype(np.uint8)
    else:
        images_np = images_np.astype(np.uint8)
    image_list = [Image.fromarray(img) for img in images_np]
    buffer = io.BytesIO()
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    image_list[0].save(
        buffer,
        format='GIF',
        save_all=True,
        append_images=image_list[1:],
        duration=100,
        loop=0,
    )
    buffer.seek(0)
    # https://ipython.readthedocs.io/en/8.27.0/api/generated/IPython.display.html
    display(IPythonImage(data=buffer.getvalue(), format='gif'))

def images_to_patches(images, patch_size=16):
    return rearrange(images, 'c (h s1) (w s2) -> (h w) (c s1 s2)', s1=patch_size, s2=patch_size)

def patches_to_images(patches, image_shape, patch_size=16):
    channels, height, width = image_shape
    patch_height = height // patch_size
    patch_width = width // patch_size
    reconstructed = rearrange(
        patches, 
        'b (ph pw) (c ps1 ps2) -> b c (ph ps1) (pw ps2)',
        ph=patch_height,
        pw=patch_width,
        ps1=patch_size,
        ps2=patch_size,
    )
    return reconstructed

def normalize_to_between_minus_one_plus_one(t: torch.Tensor):
    min_val, max_val = t.min(), t.max()
    if min_val == max_val:
        return torch.zeros_like(t)
    normalized = 2 * (t - min_val) / (max_val - min_val) - 1
    return normalized

def apply_along_dimension(func, dim, tensor):
    tensor = tensor.transpose(0, dim)
    shape = tensor.shape
    tensor = tensor.reshape(shape[0], -1)
    result = torch.stack([func(tensor[:, i]) for i in range(tensor.size(1))], dim=1)
    result = result.reshape(shape).transpose(0, dim)
    return result

def discretize(x: torch.Tensor) -> torch.Tensor:
    bins = torch.linspace(x.min(), x.max(), steps=1024)
    tokens = torch.bucketize(x, bins)
    return tokens

def undiscretize(tokens: torch.Tensor, original_min: float, original_max: float):
    bins = torch.linspace(original_min, original_max, steps=1025)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers[tokens]

def interleave(tensors):
    return torch.cat(tensors, dim=1)

def deinterleave(splits, tensors):
    return torch.split(tensors, splits, dim=1)