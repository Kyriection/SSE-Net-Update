import argparse
import os
import random
import shutil
import time
import warnings
from glob import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from representation import MPNCOV
from PIL import Image
import scipy.io as io
from collections import Counter
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from models import mpncovresnet50

from torchvision import datasets
from functions import *
from imagepreprocess import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-method', default='step', type=str,
                    help='method of learning rate')
parser.add_argument('--lr-params', default=[], dest='lr_params',nargs='*',type=float,
                    action='append', help='params of lr method')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--modeldir', default=None, type=str,
                    help='director of checkpoint')
parser.add_argument('--representation', default=None, type=str,
                    help='define the representation method')
parser.add_argument('--num-classes', default=None, type=int,
                    help='define the number of classes')
parser.add_argument('--freezed-layer', default=None, type=int,
                    help='define the end of freezed layer')
parser.add_argument('--store-model-everyepoch', dest='store_model_everyepoch', action='store_true',
                    help='store checkpoint in every epoch')
parser.add_argument('--classifier-factor', default=None, type=int,
                    help='define the multiply factor of classifier')
parser.add_argument('--benchmark', default=None, type=str,
                    help='name of dataset')
parser.add_argument('--reduction', default=0, type=int,
                    help='reduction')
best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    
    representation = MPNCOV(iterNum = 5,
                      is_sqrt = True,
                      is_vec = True,
                      input_dim = 2048,
                      dimension_reduction = None)
    
    feature = mpncovresnet50(pretrained = args.pretrained)
    fc = nn.Linear(int(256*(256+1)/2), args.num_classes)
    model = nn.Sequential(feature, representation, nn.Flatten(1,2), fc)
    print(model)
    LR = Learning_rate_generater(args.lr_method, args.lr_params, args.epochs)
    params_list = [{'params': model[0].parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay},]
    params_list.append({'params': model[1].parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay})
    params_list.append({'params': model[3].parameters(),
                        'lr': args.lr*args.classifier_factor,
                        'weight_decay': 0. if args.arch.startswith('vgg') else args.weight_decay})
    optimizer = torch.optim.SGD(params_list, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
    
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        
    cudnn.benchmark = True
    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train_transforms, val_transforms, evaluate_transforms = preprocess_strategy(args.benchmark)
    
    train_dataset = datasets.ImageFolder(
            traindir,
            train_transforms)
    val_dataset = datasets.ImageFolder(valdir, val_transforms)
    
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    if evaluate_transforms is not None:
        evaluate_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, evaluate_transforms),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    if os.path.exists(args.modeldir) is not True:
        os.mkdir(args.modeldir)
    stats_ = stats(args.modeldir, args.start_epoch)
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, LR.lr_factor, epoch)
        # train for one epoch
        trainObj, top1, top5 = train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        valObj, prec1, prec5 = validate(val_loader, model, criterion)
        # update stats
        stats_._update(trainObj, top1, top5, valObj, prec1, prec5)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        filename = []
        if args.store_model_everyepoch:
            filename.append(os.path.join(args.modeldir, 'net-epoch-%s.pth.tar' % (epoch + 1)))
        else:
            filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
        filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename)
        plot_curve(stats_, args.modeldir, True)
        data = stats_
        sio.savemat(os.path.join(args.modeldir,'stats.mat'), {'data':data})
    
    
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # compute output
        #print(input.shape)
        #print(target.shape)
        input, target = rotation(input)
        #print(input.shape)
        #print(target.shape)
        output = model(input)   
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        #if args.diversity:


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            #if model.features.module.gate is not None:
            #    print(model.features.module.gate)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  #.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
                   #data_time=data_time, loss=losses))
    return losses.avg, top1.avg, top5.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                
            input, target = rotation(input)
            # compute output
            ## modified by jiangtao xie
            if len(input.size()) > 4:# 5-D tensor
                bs, crops, ch, h, w = input.size()
                output = model(input.view(-1, ch, h, w))
                # fuse scores among all crops
                output = output.view(bs, crops, -1).mean(dim=1)
            else:
                output = model(input)
            #print(output)
            #print(target)
            loss = criterion(output, target)
            if torch.sum(torch.isnan(loss)) > 0:
                print(output)
                print(target)
            #print(loss)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top1.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Learning_rate_generater(object):
    """Generates a list of learning rate for each training epoch"""
    def __init__(self, method, params, total_epoch):
        if method == 'step':
            lr_factor, lr = self.step(params, total_epoch)
        elif method == 'log':
            lr_factor, lr = self.log(params, total_epoch)
        else:
            raise KeyError("=> undefined learning rate method '{}'" .format(method))
        self.lr_factor = lr_factor
        self.lr = lr
    def step(self, params, total_epoch):
        decrease_until = params[0]
        decrease_num = len(decrease_until)
        base_factor = 0.1
        lr_factor = torch.ones(total_epoch, dtype=torch.double)
        lr = [args.lr]
        for num in range(decrease_num):
            if decrease_until[num] < total_epoch:
                lr_factor[int(decrease_until[num])] = base_factor
        for epoch in range(1,total_epoch):
            lr.append(lr[-1]*lr_factor[epoch])
        return lr_factor, lr
    def log(self, params, total_epoch):
        params = params[0]
        left_range = params[0]
        right_range = params[1]
        np_lr = np.logspace(left_range, right_range, total_epoch)
        lr_factor = [1]
        lr = [np_lr[0]]
        for epoch in range(1, total_epoch):
            lr.append(np_lr[epoch])
            lr_factor.append(np_lr[epoch]/np_lr[epoch-1])
        if lr[0] != args.lr:
            args.lr = lr[0]
        return lr_factor, lr


def adjust_learning_rate(optimizer, lr_factor, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    groups = ['features']
    groups.append('representation')
    groups.append('classifier')
    num_group = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_factor[epoch]
        print('the learning rate is set to {0:.5f} in {1:} part'.format(param_group['lr'], groups[num_group]))
        num_group += 1


def accuracy(output, target, topk=(1,)):
    #print(output.shape)
    #print(target.shape)
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #print(target)
        if (target.dim() > 1):
            target = torch.argmax(target, 1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])
        
def rotation(input):
    batch = input.shape[0]
    target = torch.tensor(np.random.permutation([0,1,2,3] * (int(batch / 4) + 1)), device = input.device)[:batch]
    target = target.long()
    image = torch.zeros_like(input)
    image.copy_(input)
    for i in range(batch):
        image[i, :, :, :] = torch.rot90(input[i, :, :, :], target[i], [1, 2])

    return image, target

if __name__ == '__main__':
    main()

