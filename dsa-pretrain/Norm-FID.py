import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy import linalg


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

# class ReferenceDataset(data.Dataset):
#     def __init__(self, root, transform=None):
#         self.samples, self.targets = self._make_dataset(root)
#         self.transform = transform
#
#     def _make_dataset(self, root):
#         domains = os.listdir(root)
#         fnames, fnames2, labels = [], [], []
#         for idx, domain in enumerate(sorted(domains)):
#             class_dir = os.path.join(root, domain)
#             cls_fnames = listdir(class_dir)
#             fnames += cls_fnames
#             fnames2 += random.sample(cls_fnames, len(cls_fnames))
#             labels += [idx] * len(cls_fnames)
#         return list(zip(fnames, fnames2)), labels
#
#     def __getitem__(self, index):
#         fname, fname2 = self.samples[index]
#         label = self.targets[index]
#         img = Image.open(fname).convert('RGB')
#         img2 = Image.open(fname2).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#             img2 = self.transform(img2)
#         return img, img2, label
#
#     def __len__(self):
#         return len(self.targets)

def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

# def get_train_loader(root, which='source', img_size=256,
#                      batch_size=8, prob=0.5, num_workers=4):
#     print('Preparing DataLoader to fetch %s images '
#           'during the training phase...' % which)
#
#     crop = transforms.RandomResizedCrop(
#         img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
#     rand_crop = transforms.Lambda(
#         lambda x: crop(x) if random.random() < prob else x)
#
#     transform = transforms.Compose([
#         rand_crop,
#         transforms.Resize([img_size, img_size]),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                              std=[0.5, 0.5, 0.5]),
#     ])
#
#     if which == 'source':
#         dataset = ImageFolder(root, transform)
#     elif which == 'reference':
#         dataset = ReferenceDataset(root, transform)
#     else:
#         raise NotImplementedError
#
#     sampler = _make_balanced_sampler(dataset.targets)
#     return data.DataLoader(dataset=dataset,
#                            batch_size=batch_size,
#                            sampler=sampler,
#                            num_workers=num_workers,
#                            pin_memory=True,
#                            drop_last=True)

def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=0, drop_last=False):  #原num_workers=4
    # print('Preparing DataLoader form ', root, '...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = 256, 192
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


# def get_test_loader(root, img_size=256, batch_size=32,
#                     shuffle=True, num_workers=4):
#     print('Preparing DataLoader for the generation phase...')
#     transform = transforms.Compose([
#         transforms.Resize([img_size, img_size]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                              std=[0.5, 0.5, 0.5]),
#     ])
#
#     dataset = ImageFolder(root, transform)
#     return data.DataLoader(dataset=dataset,
#                            batch_size=batch_size,
#                            shuffle=shuffle,
#                            num_workers=num_workers,
#                            pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


@torch.no_grad()
def calculate_fid_given_paths(path_real,path_fake, img_size, batch_size):
    # print('Calculating FID given paths %s and %s...' % (path_real, path_fake))
    device = torch.device('cuda')
    inception = InceptionV3().eval().to(device)
    loaders = [get_eval_loader(path_real, img_size, batch_size),
               get_eval_loader(path_fake, img_size, batch_size)
               ]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, total=len(loader)):
            actv = inception(x.to(device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path_real', type=str, default='D:/DL CODE/SCTON/data/test/image')
    # parser.add_argument('--path_fake', type=str, default='result/test_1/stage3_ori')

    parser.add_argument('--path_real', type=str, default=r'D:\A_image_inpainting\code\a_second\PUT-xg2\scripts\RESULT\pvqvae_ffhq_ca5_val_e19_withoutgt\gt')
    # parser.add_argument('--path_fake', type=str, default='D:/DL CODE/CR-VTON/data/test/cloth_fake')

    # parser.add_argument('--path_real', type=str, default=r'D:\dataset\CelebA\gt')
    parser.add_argument('--path_fake', type=str, default=r'D:\A_image_inpainting\code\a_second\PUT-xg2\scripts\RESULT\pvqvae_ffhq_ca10_val_e41\rec')

    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size to use')
    args = parser.parse_args()
    fid_value = calculate_fid_given_paths(args.path_real, args.path_fake, args.img_size, args.batch_size)
    print('----------------------------FID: ', fid_value)

# python fid.py --paths PATH_REAL PATH_FAKE

# 用这个距离来衡量真实图像和生成图像的相似程度，如果FID值越小，则相似程度越高。最好情况即是FID=0，两个图像相同。
# FID值越小说明模型效果越好。
