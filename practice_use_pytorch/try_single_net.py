from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader

# load FashionMNIST data.
train_dataset = datasets.MNIST(root='runtime', transform=transforms.ToTensor(), download=True, train=True)
test_dataset = datasets.MNIST(root='runtime', transform=transforms.ToTensor(), download=True, train=False)
test_loader = DataLoader(test_dataset, batch_size=test_dataset.data.shape[0])

img_size = train_dataset.data.shape[1] * train_dataset.data.shape[2]

model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(img_size, 10), torch.nn.Softmax())
loss_fn = torch.nn.CrossEntropyLoss()

batch_size = 64
dataloader = DataLoader(train_dataset, batch_size)
optimizer = torch.optim.SGD(model.parameters(), 0.005)

test_data = train_dataset.test_data
test_label = train_dataset.test_labels


def calc_accuracy():
    model.eval()
    with torch.no_grad():
        # label is [10000]
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            print(f'correct: {correct}')
            accuracy = correct / labels.size(0)
            return accuracy


def train():
    epoch = 1
    for epoch in range(0, 100):
        for batch_idx, (batch_data, batch_label) in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()
            y = model(batch_data)
            loss = loss_fn(y, batch_label)
            loss.backward()
            optimizer.step()
            # how to do this?
            # print(f"in train data: Epoch {epoch} [Batch [{batch_idx + 1}/{len(dataloader)}], lose: {loss.item()}")
        accuracy = calc_accuracy()
        print(f"Epoch {epoch}, accuracy: {accuracy}")


if __name__ == '__main__':
    train()
