import math
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from .custom_dataset import get_custom_dataset

_device = 'cpu'

def get_model_save_path(path: str):
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runtime', path)
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.mkdir(os.path.dirname(model_save_path))
    return model_save_path
    pass


def train(model: nn.Module, save_path, epoch=50, include_mnist=False):
    """
    Args:
        epoch 训练多少代
    """
    train_data, test_data = get_custom_dataset(include_mnist=include_mnist)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    if include_mnist:
        batch_size = 64
    else:
        batch_size = 32
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for t in range(epoch):
        print(f'Epoch {t + 1}\n------------------------')
        _real_train(train_dataloader, model, loss_fn, optimizer, _device)
        _train_test(test_dataloader, model, loss_fn, _device)
    print("Done!")
    torch.save(model.state_dict(), save_path)
    print('Saved PyTorch Model State to model.pth')


def _real_train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    print_batch_size = max(math.floor(size / dataloader.batch_size / 3), 100)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        # is this need?
        optimizer.zero_grad()

        if batch % print_batch_size == 0:
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


def load_model_state(model, path) -> nn.Module:
    model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    train()
