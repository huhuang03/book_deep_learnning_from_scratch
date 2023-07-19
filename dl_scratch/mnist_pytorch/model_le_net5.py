import argparse
import os.path

import torch.optim
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from ..test.hand_draw_test import handle_draw_test_img

model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime', 'model_le_net5.pth')
if not os.path.exists(os.path.dirname(model_save_path)):
    os.mkdir(os.path.dirname(model_save_path))


class Mode1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.model(x)


device = 'cpu'

def train():
    training_data = datasets.MNIST(
        root='runtime',
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root='runtime',
        train=False,
        download=True,
        transform=ToTensor()
    )
    model = Mode1().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    epoch = 20
    for t in range(epoch):
        print(f'Epoch {t + 1}\n------------------------')
        _real_train(train_dataloader, model, loss_fn, optimizer, device)
        _train_test(test_dataloader, model, loss_fn, device)
    print("Done!")
    torch.save(model.state_dict(), model_save_path)
    print('Saved PyTorch Model State to model.pth')


def _real_train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        # is this need?
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def _train_test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, current = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            current += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    current /= size
    print(f"Test Error: \n Accuracy: {(100 * current):0.1f}%, Avg loss: {test_loss:>8f}\n")

def load_model() -> nn.Module:
    model = Mode1()
    model.load_state_dict(torch.load(model_save_path))
    return model

def test():
    model = load_model()

    def forward(x):
        x = ToTensor()(x)
        x = x.unsqueeze(0)
        x = x.to(device)
        model.eval()
        with torch.no_grad():
            y = model(x)
            return y

    handle_draw_test_img(forward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if args.t:
        test()
    else:
        train()
