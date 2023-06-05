import random
import cv2

import torch
import torchvision

dataset = torchvision.datasets.MNIST(
    root="runtime",
    train=True,
    download=True
)


random_img: torch.Tensor = dataset.data[random.randint(0, dataset.data.shape[0] - 1)]
# print(dataset.data.shape)
# print(dataset.data.size())
# print(type(random_img))
# cv2.imshow('random_img', random_img.numpy())
# cv2.waitKey(0)
cv2.imwrite('runtime/random_img.jpg', random_img.numpy())