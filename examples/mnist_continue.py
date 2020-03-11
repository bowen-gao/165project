import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=10)
parser.add_argument('--data_aug', type=eval, default=False, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--random_select', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq1111 import odeint_adjoint as odeint
else:
    from torchdiffeq1111 import odeint


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x, tol):
        self.integration_time = self.integration_time.type_as(x)
        lis, out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol)
        return lis, out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_Mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0, train_num=500, oracle_num=5000):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    ids = np.loadtxt("ids.txt", dtype=int)
    data = datasets.MNIST(root='.data/Mnist', train=True, download=True, transform=transform_test)
    random_ids = np.random.choice(6000, 600, replace=False)
    if args.random_select:
        ids = random_ids
    train_loader = DataLoader(
        datasets.MNIST(root='.data/Mnist', train=True, download=True, transform=transform_train),
        batch_size=batch_size,
        shuffle=False, num_workers=2, drop_last=False,
        sampler=torch.utils.data.SubsetRandomSampler(ids)
    )
    import collections
    dic = collections.defaultdict(int)
    for i, (inputs, targets) in enumerate(train_loader):
        dic[targets.data.tolist()[0]] += 1
    print(dic)
    train_loader_new = DataLoader(
        datasets.MNIST(root='.data/MNIST', train=True, download=True, transform=transform_train),
        batch_size=batch_size,
        shuffle=False, num_workers=2, drop_last=True,
        sampler=torch.utils.data.RandomSampler(data, replacement=True, num_samples=oracle_num)
    )
    eva = datasets.MNIST(root='.data/MNIST', train=True, download=True, transform=transform_test)
    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/MNIST', train=True, download=True, transform=transform_test),
        batch_size=100, shuffle=False, num_workers=2, drop_last=True,
        sampler=torch.utils.data.RandomSampler(eva, replacement=True, num_samples=1000)
    )
    test_loader = DataLoader(
        datasets.MNIST(root='.data/MNIST', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader, train_loader_new


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader, tol, type):
    total_correct = 0
    total_loss = 0.0
    for x, y in dataset_loader:
        x = x.to(device)
        temp_y = y.to(device)
        step_sizes, logits = model(x, tol)
        total_loss += criterion(logits, temp_y).cpu().detach().item()
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        logits = logits.cpu().detach().numpy()
        predicted_class = np.argmax(logits, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    if type == "train":
        return total_loss / len(dataset_loader), total_correct / 600
    else:
        return total_loss / len(dataset_loader), total_correct / 10000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class NODEIMG(nn.Module):

    def __init__(self):
        super(NODEIMG, self).__init__()

        self.downconv1 = nn.Conv2d(1, 64, 3, 1)
        self.downnorm1 = norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.downconv2 = nn.Conv2d(64, 64, 4, 2, 1)
        self.downnorm2 = norm(64)
        self.downconv3 = nn.Conv2d(64, 64, 4, 2, 1)
        self.odeblock = ODEBlock(ODEfunc(64))
        self.fcnorm = norm(64)
        self.adaavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcflatten = Flatten()
        self.fclinear = nn.Linear(64, 10)

    def forward(self, x, tol):
        out = self.downconv1(x)
        out = self.downnorm1(out)
        out = self.relu(out)
        out = self.downconv2(out)
        out = self.downnorm2(out)
        out = self.relu(out)
        out = self.downconv3(out)
        lis, out = self.odeblock(out, tol)
        out = self.fcnorm(out)
        out = self.relu(out)
        out = self.adaavgpool(out)
        out = self.fcflatten(out)
        out = self.fclinear(out)
        return lis, out


if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'

    model = torch.load('models/mnist_500', map_location=device)
    criterion = nn.CrossEntropyLoss().to(device)
    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))
    train_num = 500
    oracle_num = 5000
    train_loader, test_loader, train_eval_loader, train_loader_new = get_Mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size, train_num=train_num, oracle_num=oracle_num
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    tol = 1e-3
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for itr in range(args.nepochs * batches_per_epoch):
        epoch = int(itr / batches_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        step_sizes, logits = model(x, tol)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        if itr != 0 and itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_loss, train_acc = accuracy(model, train_loader, tol, "train")
                # print(train_acc)
                val_loss, val_acc = accuracy(model, test_loader, tol, "val")
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                test_loss_list.append(val_loss)
                test_acc_list.append(val_acc)
                print(train_loss, train_acc, val_loss, val_acc)
                # if val_acc > best_acc:
                #   torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
    torch.save(model, "models/mnist_new")
    np.savetxt("train_loss", train_loss_list)
    np.savetxt("train_acc", train_acc_list)
    np.savetxt("test_loss", test_loss_list)
    np.savetxt("test_acc", test_acc_list)
