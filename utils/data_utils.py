import torch
from torchvision import datasets
from torchvision import transforms
def load_data(args):
    if args.dataset == 'mnist':
        return load_mnist(args)
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

def load_batch(args, batch):
    if args.dataset == 'mnist':
        batch = batch.reshape(-1, 28*28).to(args.device)
        return batch

    return 0
