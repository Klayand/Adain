import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image
from tqdm import tqdm

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)  # clamp 用于控制 Tensor的值的大小
    return res

class TransferSet(Dataset):
    def __init__(self, content_dir, style_dir, transfroms=trans):
        content_dir_resized = content_dir + '_resized'
        style_dir_resized = style_dir + '_resized'

        if not (os.path.exists(content_dir_resized) and os.path.exists(style_dir_resized)):
            os.mkdir(content_dir_resized)
            os.mkdir(style_dir_resized)

            self._resize(content_dir, content_dir_resized)
            self._resize(style_dir, style_dir_resized)

        content_images = glob.glob((content_dir_resized + '/*'))
        style_images = glob.glob((style_dir_resized + '/*'))
        self.image_pairs = list(zip(content_images, style_images))
        self.transfroms = transfroms

    @staticmethod
    def _resize(source_dir, traget_dir):
        print(f"Starting resizing {source_dir}")

        for path in os.listdir(source_dir):
            filename = source_dir + '/' + path

            try:
                image = io.imread(filename)
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    H, W, C = image.shape
                    if H < W:
                        ratio = W / H
                        H = 512
                        W = int(ratio * H)
                    else:
                        ratio = H / W
                        W = 512
                        H = int(ratio * W)

                    # 可以放大或者缩小图像
                    image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)
                    io.imsave(os.path.join(traget_dir, path), image)
            except:
                continue

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        content_image, style_image = self.image_pairs[idx]
        content_image = Image.open(content_image)
        style_image = Image.open(style_image)

        if self.transfroms:
            content_image = self.transfroms(content_image)
            style_image = self.transfroms(style_image)

        return content_image, style_image






