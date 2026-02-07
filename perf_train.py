import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import CIFARNet
from tqdm import tqdm

def train_cifar100(model, device, epochs, batch_size, lr):
    model = model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    n_samples = 0
    start = time.perf_counter()

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for (inputs, labels) in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            n_samples += inputs.size(0)

        print(f"Epoch {epoch+1}, Loss:{running_loss/len(trainloader)}")

    total_time = time.perf_counter() - start
    _speed = n_samples / total_time
    print(f"Device {device} speed: {_speed:.1f} samples/sec")
    return _speed

if __name__ == '__main__':
    epochs = 2
    bs = 128
    lr = 0.001
    model = CIFARNet()
    cpu_speed = train_cifar100(model=model, device='cpu', epochs=epochs, batch_size=bs, lr=lr)
    if torch.cuda.is_available():
        _model = CIFARNet()
        gpu_speed = train_cifar100(model=_model, device='cuda', epochs=epochs, batch_size=bs, lr=lr)
        print(f"GPU is {gpu_speed/cpu_speed:.2f}x speed of CPU")
