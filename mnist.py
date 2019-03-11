# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import argparse
import time
import csv
import datetime
import os
import shutil

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

from utils import tensor2array

parser = argparse.ArgumentParser(description='MNIST and SVHN training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', dest='dataset', type=str, default='both',
                    help='mnist|svhn|both')
parser.add_argument('--DEBUG', action='store_true', help='DEBUG Mode')
parser.add_argument('--name', dest='name', type=str, default='demo', required=True,
                    help='name of the experiment, checpoints are stored in checpoints/name')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')

parser.add_argument('--pretrained-alice', dest='pretrained_alice', default=None, metavar='PATH',
                    help='path to pre-trained alice model')
parser.add_argument('--pretrained-bob', dest='pretrained_bob', default=None, metavar='PATH',
                    help='path to pre-trained bob model')
parser.add_argument('--pretrained-mod', dest='pretrained_mod', default=None, metavar='PATH',
                    help='path to pre-trained moderator')

parser.add_argument('--fix-alice', dest='fix_alice', action='store_true', help='do not train alicenet')
parser.add_argument('--fix-bob', dest='fix_bob', action='store_true', help='do not train bobnet')
parser.add_argument('--fix-mod', dest='fix_mod', action='store_true', help='do not train moderator')
parser.add_argument('--wr', default=1., type=float, help='moderator regularization weight')

parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--log-terminal', action='store_true', help='will display progressbar at terminal')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=0)

best_error = -1
n_iter = 0

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

def mod_regularization_loss(pred_mod):
    var_loss = torch.abs(F.sigmoid(pred_mod).var() - 0.25)
    return F.relu(var_loss-0.05)

def collaboration_loss(pred_mod, loss_alice, loss_bob):
    pseudo_label = (loss_alice < loss_bob).type_as(pred_mod)
    pseudo_label = Variable(pseudo_label.data).cuda()
    return F.binary_cross_entropy_with_logits(pred_mod.squeeze(), pseudo_label)

def init_weights(m):

    if type(m) == nn.Linear:
        torch.nn.init.normal(m.weight, mean=0, std=1)
        m.bias.data.fill_(0.01)

def save_alice_bob_mod(save_path, alice_state, bob_state, mod_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['alice', 'bob', 'mod']
    states = [alice_state, bob_state, mod_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))


def main():
    global args, best_error, n_iter
    args = parser.parse_args()

    save_path = Path(args.name)
    args.data = Path(args.data)

    args.save_path = 'checkpoints'/save_path #/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path)
    output_writer= SummaryWriter(args.save_path/'valid')

    print("=> fetching dataset")
    mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    trainset_mnist = torchvision.datasets.MNIST(args.data/'mnist', train=True, transform=mnist_transform, target_transform=None, download=True)
    valset_mnist = torchvision.datasets.MNIST(args.data/'mnist', train=False, transform=mnist_transform, target_transform=None, download=True)

    svhn_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(28,28)),
                                                    torchvision.transforms.Grayscale(),
                                                    torchvision.transforms.ToTensor()])
    trainset_svhn = torchvision.datasets.SVHN(args.data/'svhn', split='train', transform=svhn_transform, target_transform=None, download=True)
    valset_svhn = torchvision.datasets.SVHN(args.data/'svhn', split='test', transform=svhn_transform, target_transform=None, download=True)

    if args.dataset == 'mnist':
        print("Training only on MNIST")
        train_set, val_set = trainset_mnist, valset_mnist
    elif args.dataset == 'svhn':
        print("Training only on SVHN")
        train_set, val_set = trainset_svhn, valset_svhn
    else:
        print("Training on both MNIST and SVHN")
        train_set = torch.utils.data.ConcatDataset([trainset_mnist, trainset_svhn])
        val_set = torch.utils.data.ConcatDataset([valset_mnist, valset_svhn])

    print('{} Train samples and {} test samples found in MNIST'.format(len(trainset_mnist), len(valset_mnist)))
    print('{} Train samples and {} test samples found in SVHN'.format(len(trainset_svhn), len(valset_svhn)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    alice_net = LeNet()
    bob_net = LeNet()
    mod_net = LeNet(nout=1)

    if args.pretrained_alice:
        print("=> using pre-trained weights from {}".format(args.pretrained_alice))
        weights = torch.load(args.pretrained_alice)
        alice_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_bob:
        print("=> using pre-trained weights from {}".format(args.pretrained_bob))
        weights = torch.load(args.pretrained_bob)
        bob_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_mod:
        print("=> using pre-trained weights from {}".format(args.pretrained_mod))
        weights = torch.load(args.pretrained_mod)
        mod_net.load_state_dict(weights['state_dict'], strict=False)

    if args.resume:
        print("=> resuming from checkpoint")
        alice_weights = torch.load(args.save_path/'alicenet_checkpoint.pth.tar')
        bob_weights = torch.load(args.save_path/'bobnet_checkpoint.pth.tar')
        mod_weights = torch.load(args.save_path/'modnet_checkpoint.pth.tar')

        alice_net.load_state_dict(alice_weights['state_dict'])
        bob_net.load_state_dict(bob_weights['state_dict'])
        mod_net.load_state_dict(mod_weights['state_dict'])

    cudnn.benchmark = True
    alice_net = alice_net.cuda()
    bob_net = bob_net.cuda()
    mod_net = mod_net.cuda()

    print('=> setting adam solver')

    parameters = chain(alice_net.parameters(), bob_net.parameters(), mod_net.parameters())
    optimizer_compete = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    optimizer_collaborate = torch.optim.Adam(mod_net.parameters(), args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)


    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['val_loss_full', 'val_loss_alice', 'val_loss_bob'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss_full', 'train_loss_alice', 'train_loss_bob'])

    if args.log_terminal:
        logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
        logger.epoch_bar.start()
    else:
        logger=None

    for epoch in range(args.epochs):
        mode = 'compete' if (epoch%2)==0 else 'collaborate'

        if args.fix_alice:
            for fparams in alice_net.parameters():
                fparams.requires_grad = False

        if args.fix_bob:
            for fparams in bob_net.parameters():
                fparams.requires_grad = False

        if args.fix_mod:
            mode = 'compete'
            for fparams in mod_net.parameters():
                fparams.requires_grad = False

        if args.log_terminal:
            logger.epoch_bar.update(epoch)
            logger.reset_train_bar()

        # train for one epoch
        if mode == 'compete':
            train_loss = train(train_loader, alice_net, bob_net, mod_net, optimizer_compete, args.epoch_size, logger, training_writer, mode=mode)
        elif mode == 'collaborate':
            train_loss = train(train_loader, alice_net, bob_net, mod_net, optimizer_collaborate, args.epoch_size, logger, training_writer, mode=mode)

        if args.log_terminal:
            logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))
            logger.reset_valid_bar()

        if epoch%1==0:

            # evaluate on validation set
            errors, error_names = validate(val_loader, alice_net, bob_net, mod_net, epoch, logger, output_writer)

            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))

            if args.log_terminal:
                logger.valid_writer.write(' * Avg {}'.format(error_string))
            else:
                print('Epoch {} completed'.format(epoch))

            for error, name in zip(errors, error_names):
                training_writer.add_scalar(name, error, epoch)

            # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)

            if args.fix_alice:
                decisive_error = errors[2]
            elif args.fix_bob:
                decisive_error = errors[1]
            else:
                decisive_error = errors[0]     # epe_total
            if best_error < 0:
                best_error = decisive_error

            # remember lowest error and save checkpoint
            is_best = decisive_error <= best_error
            best_error = min(best_error, decisive_error)
            save_alice_bob_mod(
                args.save_path, {
                    'epoch': epoch + 1,
                    'state_dict': alice_net.state_dict()
                }, {
                    'epoch': epoch + 1,
                    'state_dict': bob_net.state_dict()
                }, {
                    'epoch': epoch + 1,
                    'state_dict': mod_net.state_dict()
                },
                is_best)

            with open(args.save_path/args.log_summary, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow([train_loss, decisive_error])

    if args.log_terminal:
        logger.epoch_bar.finish()


def train(train_loader, alice_net, bob_net, mod_net, optimizer, epoch_size, logger=None, train_writer=None, mode='compete'):
    global args, n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    # switch to train mode
    alice_net.train()
    bob_net.train()
    mod_net.train()

    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        # measure data loading time
        #mode = 'compete' if (i%2)==0 else 'collaborate'

        data_time.update(time.time() - end)
        img_var = Variable(img.cuda())
        target_var = Variable(target.cuda())

        pred_alice = alice_net(img_var)
        pred_bob = bob_net(img_var)
        pred_mod = mod_net(img_var)

        loss_alice = F.cross_entropy(pred_alice, target_var, reduce=False)
        loss_bob = F.cross_entropy(pred_bob, target_var, reduce=False)

        if mode=='compete':
            if args.fix_bob:
                if args.DEBUG: print("Training Alice Only")
                loss = loss_alice.mean()
            elif args.fix_alice:
                loss = loss_bob.mean()
            else:
                if args.DEBUG: print("Training Both Alice and Bob")

                pred_mod_soft = Variable(F.sigmoid(pred_mod).data, requires_grad=False)
                loss = pred_mod_soft*loss_alice + (1-pred_mod_soft)*loss_bob

                loss = loss.mean()

        elif mode=='collaborate':
            loss_alice2 = Variable(loss_alice.data, requires_grad = False)
            loss_bob2 = Variable(loss_bob.data, requires_grad = False)

            loss1 = F.sigmoid(pred_mod)*loss_alice2 + (1-F.sigmoid(pred_mod))*loss_bob2

            loss2 = collaboration_loss(pred_mod, loss_alice2, loss_bob2)

            loss = loss1.mean() + loss2.mean() + args.wr*mod_regularization_loss(pred_mod)


        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('loss_alice', loss_alice.mean().item(), n_iter)
            train_writer.add_scalar('loss_bob', loss_bob.mean().item(), n_iter)
            train_writer.add_scalar('mod_mean', F.sigmoid(pred_mod).mean().item(), n_iter)
            train_writer.add_scalar('mod_var', F.sigmoid(pred_mod).var().item(), n_iter)
            train_writer.add_scalar('loss_regularization', mod_regularization_loss(pred_mod).item(), n_iter)

            if mode=='compete':
                train_writer.add_scalar('competetion_loss', loss.item(), n_iter)
            elif mode=='collaborate':
                train_writer.add_scalar('collaboration_loss', loss.item(), n_iter)

        # record loss
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_alice.mean().item(), loss_bob.mean().item()])
        if args.log_terminal:
            logger.train_bar.update(i+1)
            if i % args.print_freq == 0:
                logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


def validate(val_loader, alice_net, bob_net, mod_net, epoch, logger=None, output_writer=[]):
    global args
    batch_time = AverageMeter()
    accuracy = AverageMeter(i=3, precision=4)

    # switch to evaluate mode
    alice_net.eval()
    bob_net.eval()
    mod_net.eval()

    end = time.time()

    for i, (img, target) in enumerate(val_loader):
        img_var = Variable(img.cuda(), volatile=True)
        target_var = Variable(target.cuda(), volatile=True)

        pred_alice = alice_net(img_var)
        pred_bob = bob_net(img_var)
        pred_mod = F.sigmoid(mod_net(img_var))

        _ , pred_alice_label = torch.max(pred_alice.data, 1)
        _ , pred_bob_label = torch.max(pred_bob.data, 1)
        pred_label = (pred_mod.squeeze().data > 0.5).type_as(pred_alice_label) * pred_alice_label + (pred_mod.squeeze().data <= 0.5).type_as(pred_bob_label) * pred_bob_label

        total_accuracy = (pred_label.cpu() == target).sum() / img.size(0)
        alice_accuracy = (pred_alice_label.cpu() == target).sum() / img.size(0)
        bob_accuracy = (pred_bob_label.cpu() == target).sum() / img.size(0)

        accuracy.update([total_accuracy, alice_accuracy, bob_accuracy])


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.log_terminal:
            logger.valid_bar.update(i)
            if i % args.print_freq == 0:
                logger.valid_writer.write('valid: Time {} Accuray {}'.format(batch_time, accuracy))

    if args.log_output:
        output_writer.add_scalar('accuracy_alice', accuracy.avg[1], epoch)
        output_writer.add_scalar('accuracy_bob', accuracy.avg[2], epoch)
        output_writer.add_scalar('accuracy_total', accuracy.avg[0], epoch)

    if args.log_terminal:
        logger.valid_bar.update(len(val_loader))

    return list(map(lambda x: 1-x, accuracy.avg)), ['Total loss', 'alice loss', 'bob loss']



if __name__ == '__main__':
    import sys
    with open("experiment_recorder.md", "a") as f:
        f.write('\n python3 ' + ' '.join(sys.argv))
    main()
