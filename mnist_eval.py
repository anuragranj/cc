# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import argparse
import time
import csv
import datetime
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.functional as F

from logger import TermLogger, AverageMeter
from path import Path
from itertools import chain
from tensorboardX import SummaryWriter

from utils import tensor2array, save_checkpoint

parser = argparse.ArgumentParser(description='MNIST and SVHN training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size')

parser.add_argument('--pretrained-alice', dest='pretrained_alice', default=None, metavar='PATH',
                    help='path to pre-trained alice model')
parser.add_argument('--pretrained-bob', dest='pretrained_bob', default=None, metavar='PATH',
                    help='path to pre-trained bob model')
parser.add_argument('--pretrained-mod', dest='pretrained_mod', default=None, metavar='PATH',
                    help='path to pre-trained moderator')

class LeNet(nn.Module):
    def __init__(self, nout=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, 3, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.fc1 = nn.Linear(40*5*5, 40)
        self.fc2 = nn.Linear(40, nout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 40*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"

def main():
    global args
    args = parser.parse_args()

    args.data = Path(args.data)

    print("=> fetching dataset")
    mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    valset_mnist = torchvision.datasets.MNIST(args.data/'mnist', train=False, transform=mnist_transform, target_transform=None, download=True)

    svhn_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(28,28)),
                                                    torchvision.transforms.Grayscale(),
                                                    torchvision.transforms.ToTensor()])
    valset_svhn = torchvision.datasets.SVHN(args.data/'svhn', split='test', transform=svhn_transform, target_transform=None, download=True)
    val_set = torch.utils.data.ConcatDataset([valset_mnist, valset_svhn])


    print('{} Test samples found in MNIST'.format(len(valset_mnist)))
    print('{} Test samples found in SVHN'.format(len(valset_svhn)))

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    val_loader_mnist = torch.utils.data.DataLoader(
        valset_mnist, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    val_loader_svhn = torch.utils.data.DataLoader(
        valset_svhn, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    # create model
    print("=> creating model")

    alice_net = LeNet()
    bob_net = LeNet()
    mod_net = LeNet(nout=1)

    print("=> using pre-trained weights from {}".format(args.pretrained_alice))
    weights = torch.load(args.pretrained_alice)
    alice_net.load_state_dict(weights['state_dict'])

    print("=> using pre-trained weights from {}".format(args.pretrained_bob))
    weights = torch.load(args.pretrained_bob)
    bob_net.load_state_dict(weights['state_dict'])

    print("=> using pre-trained weights from {}".format(args.pretrained_mod))
    weights = torch.load(args.pretrained_mod)
    mod_net.load_state_dict(weights['state_dict'])

    cudnn.benchmark = True
    alice_net = alice_net.cuda()
    bob_net = bob_net.cuda()
    mod_net = mod_net.cuda()

    # evaluate on validation set
    errors_mnist, error_names_mnist, mod_count_mnist = validate(val_loader_mnist, alice_net, bob_net, mod_net)
    errors_svhn, error_names_svhn, mod_count_svhn = validate(val_loader_svhn, alice_net, bob_net, mod_net)
    errors_total, error_names_total, _ = validate(val_loader, alice_net, bob_net, mod_net)

    accuracy_string_mnist = ', '.join('{} : {:.3f}'.format(name, 100*(error)) for name, error in zip(error_names_mnist, errors_mnist))
    accuracy_string_svhn = ', '.join('{} : {:.3f}'.format(name, 100*(error)) for name, error in zip(error_names_svhn, errors_svhn))
    accuracy_string_total = ', '.join('{} : {:.3f}'.format(name, 100*(error)) for name, error in zip(error_names_total, errors_total))

    print("MNIST Error")
    print(accuracy_string_mnist)
    print("MNIST Picking Percentage- Alice {:.3f}, Bob {:.3f}".format(mod_count_mnist[0]*100, (1-mod_count_mnist[0])*100))

    print("SVHN Error")
    print(accuracy_string_svhn)
    print("SVHN Picking Percentage for Alice {:.3f}, Bob {:.3f}".format(mod_count_svhn[0]*100, (1-mod_count_svhn[0])*100))

    print("TOTAL Error")
    print(accuracy_string_total)

def validate(val_loader, alice_net, bob_net, mod_net):
    global args
    accuracy = AverageMeter(i=3, precision=4)
    mod_count = AverageMeter()

    # switch to evaluate mode
    alice_net.eval()
    bob_net.eval()
    mod_net.eval()

    for i, (img, target) in enumerate(tqdm(val_loader)):
        img_var = Variable(img.cuda(), volatile=True)
        target_var = Variable(target.cuda(), volatile=True)

        pred_alice = alice_net(img_var)
        pred_bob = bob_net(img_var)
        pred_mod = F.sigmoid(mod_net(img_var))
        _ , pred_alice_label = torch.max(pred_alice.data, 1)
        _ , pred_bob_label = torch.max(pred_bob.data, 1)
        pred_label = (pred_mod.squeeze().data > 0.5).type_as(pred_alice_label) * pred_alice_label + (pred_mod.squeeze().data <= 0.5).type_as(pred_bob_label) * pred_bob_label

        total_accuracy = (pred_label.cpu() == target).sum().item() / img.size(0)
        alice_accuracy = (pred_alice_label.cpu() == target).sum().item() / img.size(0)
        bob_accuracy = (pred_bob_label.cpu() == target).sum().item() / img.size(0)
        accuracy.update([total_accuracy, alice_accuracy, bob_accuracy])
        mod_count.update((pred_mod.cpu().data > 0.5).sum().item() / img.size(0))

    return list(map(lambda x: 1-x, accuracy.avg)), ['Total', 'alice', 'bob'] , mod_count.avg



if __name__ == '__main__':
    # import sys
    # with open("experiment_recorder.md", "a") as f:
    #     f.write('\n python3 ' + ' '.join(sys.argv))
    main()
