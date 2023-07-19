import cv2
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToPILImage

training_data = datasets.MNIST(
    root='runtime',
    train=True,
    download=True,
    transform=ToPILImage()
)

random = np.random.randint(0, training_data.data.shape[0])
print(random)


cv2.imshow('img', np.asarray(training_data.data[random]))
cv2.waitKey(0)