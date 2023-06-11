import cv2
from torchvision import datasets, transforms
import torch

# load FashionMNIST data.
train_dataset = datasets.MNIST(root='runtime', transform=transforms.ToTensor(), download=True, train=True)
imgs = train_dataset.data.numpy()
img0 = train_dataset.data[0].numpy()

height, width = img0.shape
img_size = height * width

# 网络
# Line -> Relu -> Line -> Softmax
# lost: CrossEntropyCross
# loss_fn = torch.nn.CrossEntropyLoss()

# class TwoLayerNet:
#     def __init__(self):
#         self.flatten = torch.nn.Flatten()
#         self.nets = torch.nn.Sequential([
#             torch.nn.Linear(img_size, 10)
#         ])
#         # 首先全联接
#         pass
