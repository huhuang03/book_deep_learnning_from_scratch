import torch.nn
from torch import nn

from .model_util import train, get_model_save_path, load_model_state

model_save_path = get_model_save_path('module_simple_conv.pth')


class ModuleSimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 30, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2),
            nn.Flatten(),
            nn.Linear(4320, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        return self.model(x)


def load_module() -> torch.nn.Module:
    m = ModuleSimpleConv()
    load_model_state(m, model_save_path)
    return m


if __name__ == '__main__':
    model = ModuleSimpleConv()
    train(model, model_save_path, epoch=30)
