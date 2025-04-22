import torch
import torchvision
import torchvision.transforms as transforms


def load_cifar10_data(data_dir='./data', batch_size=32):
    """
    Load and preprocess the CIFAR-10 dataset with data augmentation for training.

    Args:
        data_dir (str): Directory to store the dataset.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        tuple: (trainloader, testloader) - DataLoaders for training and test datasets.
    """
    # Define transform for training data with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Randomly rotate the image by up to 10 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    # Define transform for test data (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    # Load training dataset with augmentation
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load test dataset without augmentation
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


if __name__ == "__main__":
    trainloader, testloader = load_cifar10_data()
    print(f"Training data shape: {trainloader.dataset.data.shape}")
    print(f"Test data shape: {testloader.dataset.data.shape}")
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(f"Sample image mean: {images.mean().item()}")
    print(f"Sample image std: {images.std().item()}")