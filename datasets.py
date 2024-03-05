import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x if (x.ndim == 3) and (x.shape[0] == 3) else x.repeat(3, 1, 1)),
    transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def get_mnist():
    return {
        'train': torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform),
        'test': torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    }

def get_cifar():
    return {
        'train': torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
        'test': torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    }