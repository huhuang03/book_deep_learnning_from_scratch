# how to do the custom dataset?
import os.path
import random
from typing import List, Any, Tuple

import cv2
import torch
from torchvision.datasets import MNIST
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor

to_tensor = ToTensor()


def get_custom_dataset(include_mnist=True, train_percent=0.2) -> [VisionDataset, VisionDataset]:
    """
    Args:
        train_percent how many train_percent of the all dataset

    Returns:
         [train_dataset, test_dataset]
    """
    mnist_train = None
    mnist_test = None

    if include_mnist:
        mnist_root = 'runtime/mnist'
        mnist_train = MNIST(
            mnist_root,
            train=True,
            download=True,
            transform=ToTensor()
        )
        mnist_test = MNIST(
            mnist_root,
            train=False,
            transform=ToTensor()
        )
    manually_items = _load_manually_items()
    random.shuffle(manually_items)
    manually_trains_len = int(len(manually_items) * (1 - train_percent))
    manually_trains = manually_items[:manually_trains_len]
    manually_tests = manually_items[manually_trains_len:]
    return MixedDataset(mnist_train, ManuallyDataset(manually_trains)),\
        MixedDataset(mnist_test, ManuallyDataset(manually_tests))


manually_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runtime', 'dataset')


def _load_manually_items() -> List['ManuallyItem']:
    manually_items = []
    for f in os.listdir(manually_root):
        manually_items.append(ManuallyItem.create(os.path.join(manually_root, f)))
    return manually_items


class MixedDataset(VisionDataset):
    """
    this mixed manually and mnist
    """

    def __init__(self, mnist: VisionDataset | None, manually: VisionDataset):
        super().__init__('')
        self.mnist = mnist
        self.mnist_len = len(mnist) if mnist is not None else 0
        self.manually = manually

    def __len__(self):
        return self.mnist_len + len(self.manually)

    def __getitem__(self, index):
        if index < self.mnist_len:
            return self.mnist[index]
        return self.manually[index - self.mnist_len]


class ManuallyDataset(VisionDataset):
    def __init__(self, items: List['ManuallyItem']):
        super().__init__('')
        self.items = items

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.items[index].img, self.items[index].target

    def __len__(self) -> int:
        return len(self.items)


class ManuallyItem:
    def __init__(self, data: torch.Tensor, target: int):
        """
        Args:
            data: has normalized image
        """
        super().__init__()
        self.img = data
        self.target = target
        assert self.img is not None
        assert self.target is not None and 0 <= self.target <= 9

    @staticmethod
    def create(path: str) -> 'ManuallyItem':
        name: str = os.path.basename(path)
        target = int(name[name.rindex('_') + 1:name.rindex('.jpg')])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = to_tensor(img)
        assert target is not None and 0 <= target <= 9
        return ManuallyItem(img, target)
