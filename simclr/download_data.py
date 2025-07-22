# data.py

# import cifar10
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])

train_data = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)