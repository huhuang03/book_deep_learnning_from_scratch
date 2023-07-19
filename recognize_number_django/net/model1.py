from torch import nn

from .model_util import train, get_model_save_path, load_model_state

model_save_path = get_model_save_path('model1.pth')

def load_model():
    m = Model1()
    load_model_state(m, model_save_path)
    return m

class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10),
        )

    def forward(self, x):
        return self.model.forward(x)


if __name__ == '__main__':
    model = Model1()
    train(model, model_save_path)