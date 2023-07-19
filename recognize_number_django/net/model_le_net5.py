import torch.nn
from torch import nn

from .model_util import train, get_model_save_path, load_model_state

model_save_path = get_model_save_path('module_le_net5.pth')


class ModuleLeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.model(x)


def load_module() -> torch.nn.Module:
    m = ModuleLeNet5()
    load_model_state(m, model_save_path)
    return m


if __name__ == '__main__':
    model = ModuleLeNet5()
    train(model, model_save_path, include_mnist=False, epoch=40)
