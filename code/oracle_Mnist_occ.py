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
import scipy.misc
from scipy import ndimage
import matplotlib.pyplot as plt
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=10)
parser.add_argument('--data_aug', type=eval, default=False, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--oracle_size', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--noise_std', type=float, default=0)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


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

    data = datasets.MNIST(root='.data/MNIST', train=True, download=True, transform=transform_test)
    train_loader = DataLoader(
        datasets.MNIST(root='.data/MNIST', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=False, num_workers=2, drop_last=True,
        sampler=torch.utils.data.RandomSampler(data, replacement=True, num_samples=train_num)
    )

    train_index = np.loadtxt('indices_mnist.txt')
    index = np.arange(50000)
    temp = np.delete(index, train_index)
    oracle_index = np.random.choice(temp, args.oracle_size, replace=False)
    train_loader_new = DataLoader(
        datasets.MNIST(root='.data/MNIST', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=False, num_workers=2, drop_last=True,
        sampler=torch.utils.data.SubsetRandomSampler(oracle_index)
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/MNIST', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
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


def accuracy(model, dataset_loader, tol):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        _, temp = model(x, tol)
        temp = temp.cpu().detach().numpy()
        predicted_class = np.argmax(temp, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


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


def addnoise(x, occlude_size):
    M = x.size(2)
    N = x.size(3)
    med_M = int(M / 2)
    med_N = int(N / 2)
    x_noise = x
    x_noise[0, 0, med_M - int(occlude_size / 2):med_M + int(occlude_size),
    med_N - int(occlude_size / 2):med_N + int(occlude_size)] = 0
    return x_noise


if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'

    '''
    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(3, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(3, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

    feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))
    '''
    # model = NODEIMG()
    model = torch.load("models/mnist", map_location=device)
    criterion = nn.CrossEntropyLoss().to(device)
    train_num = 500
    oracle_num = 5000
    train_loader, test_loader, train_eval_loader, oracle_loader = get_Mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size, train_num=train_num, oracle_num=oracle_num
    )

    # data_gen = inf_generator(oracle_loader)
    batches_per_epoch = len(oracle_loader)

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

    print('new')
    stepnum_list = [[[] for j in range(10)] for i in range(4)]
    loss_list = [[[] for j in range(10)] for i in range(4)]
    tol = args.tol / 100000
    noise = [0, 6, 8, 10]
    with torch.no_grad():
        a=[]
        b=[]
        for i, (x, y) in enumerate(oracle_loader):
            print(i)
            # x = images.cuda(async=True)
            # y = labels.cuda(async=True)
            x = x.to(device)

            y = y.to(device)

            # img = x.detach().cpu().numpy()
            # img = np.squeeze(img)
            # img = np.moveaxis(img, [0, 1, 2], [-1, -3, -2])
            # plt.imshow(img)
            # plt.savefig('oracle/' + '_label' + str(y.detach().item()) + 'num' + str(i) + '.png')
            '''
            step_sizes, logits = model(x, tol)
            a.append(len(step_sizes))
            print("no:", len(step_sizes),np.mean(a))
            x_noise = addnoise(x, 8)
            step_sizes, logits = model(x_noise, tol)
            b.append(len(step_sizes))
            print("add:", len(step_sizes),np.mean(b))
            '''

            for i, n in enumerate(noise):
                x_noise = addnoise(x, n).to(device)
                step_sizes, logits = model(x_noise, tol)
                stepnum_list[i][y].append(len(step_sizes))
                # print(stepnum[y])
                #loss = criterion(logits, y)
                #loss_list[i][y].append(loss.detach().item())

        with open("oracle/occ_stepnum.txt", 'w') as f:
            for i in range(len(noise)):
                for y in range(len(stepnum_list[i])):
                    f.write(str(stepnum_list[i][y])[1:-1] + "\n")
        '''
        with open("oracle/loss.txt", 'w') as f:
            for i in range(len(noise)):
                for y in range(len(loss_list[i])):
                    f.write(str(loss_list[i][y])[1:-1] + "\n")

        '''
        ''' 
        for y in range(10):
            plt.hist(stepnum[y])
            if args.noise_std!=0:
                plt.hist(stepnum_noise[y])
            plt.savefig('hist/hist_'+str(y))
            plt.clf()
        '''

'''
    for itr in range(args.nepochs * batches_per_epoch):

        print('iter = ', itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)

        if itr / train_num < 1:
            img = x.detach().numpy()
            img = np.squeeze(img)
            img = np.moveaxis(img, [0, 1, 2], [-1, -3, -2])
            plt.imshow(img)
            plt.savefig('train/' + 'num' + str(itr) + '_label' + str(y.detach().item()) + '.png')

        y = y.to(device)
        step_sizes, logits = model(x, tol)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()
        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0
        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader, tol)
                val_acc = accuracy(model, test_loader, tol)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc
                    )
                )

'''
