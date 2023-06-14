from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader

# load FashionMNIST data.
train_dataset = datasets.MNIST(root='runtime', transform=transforms.ToTensor(), download=True, train=True)
imgs = train_dataset.data.numpy()
img0 = train_dataset.data[0].numpy()

height, width = img0.shape
img_size = height * width

# model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(img_size, 10), torch.nn.Softmax())
model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(img_size, 10))
loss_fn = torch.nn.CrossEntropyLoss()


batch_size = 64
dataloader = DataLoader(train_dataset, batch_size)
optimizer = torch.optim.SGD(model.parameters(), 0.005)

model.train()
for batch_idx, (batch_data, batch_label) in enumerate(dataloader):
    optimizer.zero_grad()
    y = model(batch_data)
    loss = loss_fn(y, batch_label)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(y, 1)
    correct = (predicted == batch_label).sum().item()
    accuracy = correct / batch_label.size(0)
    print(f"in train data: Epoch [Batch [{batch_idx + 1}/{len(dataloader)}], Accuracy: {accuracy:.4f}")

# 网络
# Line -> Relu -> Line -> Softmax
# lost: CrossEntropyCross

# class TwoLayerNet:
#     def __init__(self):
#         self.flatten = torch.nn.Flatten()
#         self.nets = torch.nn.Sequential([
#             torch.nn.Linear(img_size, 10)
#         ])
#         # 首先全联接
#         pass
