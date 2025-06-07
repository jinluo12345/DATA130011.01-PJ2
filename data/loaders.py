"""
Data loaders
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets
import random


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self):
        return self.dataset.__getitem__()

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(root='../data/', 
                     batch_size=128,
                     train=True,
                     shuffle=True,
                     num_workers=4,
                     n_items=-1,
                     augmentation_cfg=None):
    """
    CIFAR-10 DataLoader with dynamic augmentations.
    """
    # -------------------------------
    # Define list of dynamic transforms
    # -------------------------------
    transform_list = []

    if augmentation_cfg is None:
        augmentation_cfg = {}

    aug = augmentation_cfg  # shortcut

    # Random crop
    if aug.get('random_crop', {}).get('enabled', False):
        p = aug['random_crop'].get('p', 1.0)
        if random.random() < p:
            crop_size = aug['random_crop'].get('crop_size', 32)
            crop_pad = aug['random_crop'].get('crop_padding', 4)
            transform_list.append(transforms.RandomCrop(crop_size, padding=crop_pad))

    # Horizontal flip
    if aug.get('horizontal_flip', {}).get('enabled', False):
        p = aug['horizontal_flip'].get('p', 0.5)
        transform_list.append(transforms.RandomHorizontalFlip(p=p))

    # Random rotation
    if aug.get('random_rotation', {}).get('enabled', False):
        p = aug['random_rotation'].get('p', 1.0)
        if random.random() < p:
            deg = aug['random_rotation'].get('rotation_degree', 15)
            transform_list.append(transforms.RandomRotation(degrees=deg))

    # Random affine
    if aug.get('random_affine', {}).get('enabled', False):
        p = aug['random_affine'].get('p', 1.0)
        if random.random() < p:
            deg = aug['random_affine'].get('affine_deg', 10)
            translate = aug['random_affine'].get('affine_translate', [0.1, 0.1])
            scale = aug['random_affine'].get('affine_scale', [0.9, 1.1])
            shear = aug['random_affine'].get('affine_shear', 5)
            transform_list.append(
                transforms.RandomAffine(degrees=deg,
                                        translate=translate,
                                        scale=scale,
                                        shear=shear)
            )

    # Color jitter
    if aug.get('color_jitter', {}).get('enabled', False):
        p = aug['color_jitter'].get('p', 0.5)
        if random.random() < p:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=aug['color_jitter'].get('brightness', 0),
                    contrast=aug['color_jitter'].get('contrast', 0),
                    saturation=aug['color_jitter'].get('saturation', 0),
                    hue=aug['color_jitter'].get('hue', 0)
                )
            )

    # Grayscale
    if aug.get('random_grayscale', {}).get('enabled', False):
        p = aug['random_grayscale'].get('p', 0.1)
        transform_list.append(transforms.RandomGrayscale(p=aug['random_grayscale'].get('grayscale_prob', 0.1)))

    # ToTensor & Normalize must always be applied at the end
    transform_list.append(transforms.ToTensor())
    mean = aug.get('normalize_mean', [0.4914, 0.4822, 0.4465])
    std = aug.get('normalize_std', [0.247, 0.243, 0.261])
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    # Compose all transforms
    transform = transforms.Compose(transform_list)

    # Load CIFAR10 dataset
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

    # Use partial dataset if n_items is specified
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    # Create DataLoader
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers)

    return loader

if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break