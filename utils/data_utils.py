import torch
from torchvision import datasets
from torchvision import transforms
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

def load_data(args):
    if args.dataset == 'mnist':
        return load_mnist(args)
    elif args.dataset == 'toy':
        return load_swiss(args)
    return 0

def load_mnist(args):
    tensor_transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=tensor_transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=tensor_transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size,
        shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size,
        shuffle=True, num_workers=0)
    
    return (train_loader, test_loader)

def load_swiss(args):
    xnp, _ = make_swiss_roll(2000, noise=1.0)
    xtns = torch.as_tensor(xnp[:, [0, 2]] / 10.0, dtype=torch.float32)
    plt.plot(xtns[:, 0], xtns[:, 1], 'C0.')
    plt.savefig("swiss_roll.jpg")
    dset = torch.utils.data.TensorDataset(xtns)
    train_set, val_set = torch.utils.data.random_split(dset, [1500, 500])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
        shuffle=True, num_workers=0)
    
    return (train_loader, test_loader)

def load_batch(args, batch):
    if args.dataset == 'mnist':
        batch = batch.reshape(-1, 28*28).to(args.device)
        return batch
    elif args.dataset == 'toy':
        return batch[0].to(args.device)
    return 0
