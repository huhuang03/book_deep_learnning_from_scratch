import numpy as np
import torch
from torchvision.transforms import ToTensor
import sdl_x
import cv2


def reg(img: np.ndarray, model) -> np.ndarray:
    img = cv2.blur(img, (10, 10))
    img = ToTensor()(img)
    img = img.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        y: torch.Tensor = model(img)
        y = y.numpy()
        y = sdl_x.softmax(y)
        print(y)
    return y