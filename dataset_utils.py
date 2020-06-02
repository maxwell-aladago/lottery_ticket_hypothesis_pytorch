from torchvision import datasets, transforms
from torch.utils import data
from torch import load


def get_dataset(batch_size=60, dataset_name="mnist", augmentation=False):
    """
    construct data loaders for the training, validation and test sets. The validation set is a random subset of the
    original training set. The new training set is a compliment of the validation set.

    For MNIST, the training and validation sets have 50000 and 10000 examples respectively
    For CIFAR10, the training and validation sets have 45000 and 5000 examples respectively.
    The test sets for both MNIST and CIFAR10 has 10000 examples each.

     Arguments
     --------
    batch_size: (int) The mini batch size for gradient descent and tests. Default: 60
    dataset_name: (string) The name of the dataset. Currently supported sets are {mnist, cifar10}. Default: mnist
    augmentation: (bool) Indicates whether to perform horizontal flipping and normalization for cifar10 or not.
                No data augmentation is applied to MNIST. Default: False
    return:
    train_dl: (DataLoader) The training set data loader. This is a random subset of the original training set of
            the specified dataset
    val_dl: (DataLoader): The validation set data loader. This is a random subset of the original training
    test_dl: (DataLoader): The test set data loader of the specified dataset
    """
    if dataset_name == "mnist":
        train_dl = data.DataLoader(
            data.Subset(
                datasets.MNIST(root="./data/", train=True, download=True, transform=transforms.ToTensor()),
                load("./data/mnist_train_indices.pt")
            ),

            batch_size=batch_size,
            shuffle=True,
            num_workers=3
        )

        val_dl = data.DataLoader(
            data.Subset(
                datasets.MNIST(root="./data/", train=True, download=True, transform=transforms.ToTensor()),
                load("./data/mnist_validation_indices.pt")
            ),

            batch_size=batch_size,
            shuffle=False
        )
        test_dl = data.DataLoader(
            datasets.MNIST(root="./data/", train=False, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size,
            shuffle=False
        )
    elif dataset_name == 'cifar10':
        train_transform = transforms.ToTensor()
        test_transform = transforms.ToTensor()

        if augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            # Do not augment test and validation dataset
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        train_dl = data.DataLoader(
            data.Subset(
                datasets.CIFAR10(root="./data/", train=True, download=True, transform=train_transform),
                load("./data/cifar10_train_indices.pt")
            ),

            batch_size=batch_size,
            shuffle=True
        )

        val_dl = data.DataLoader(
            data.Subset(
                datasets.CIFAR10(root="./data/", train=True, download=True, transform=test_transform),
                load("./data/cifar10_validation_indices.pt")
            ),

            batch_size=batch_size,
            shuffle=False
        )
        test_dl = data.DataLoader(
            datasets.CIFAR10(root="./data/", train=False, download=True, transform=test_transform),
            batch_size=batch_size,
            shuffle=False
        )
    else:
        raise ValueError(f"unknown dataset {dataset_name}")

    return train_dl, val_dl, test_dl
