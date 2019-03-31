# -*-coding:utf-8-*-

import torch
import time
import argparse
import warnings
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from module import SphereFace20, AngularSoftmaxWithLoss
from utils import visual_feature_space, create_gif

warnings.filterwarnings("ignore")


class Options():
    """Configure train model parameters."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='SphereFace')
        self.parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                                 help='input batch size for training (default: 64)')
        self.parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                                 help='input batch size for testing (default: 100)')
        self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                 help='how many batches to wait before logging training status')
        self.parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                                 help='learning rate (default: 1e-3)')
        self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                 help='SGD momentum (default: 0.5)')
        self.parser.add_argument('--weight-decay', type=float, default=1e-5, metavar='W',
                                 help='SGD weight decay (default: 1e-5)')
        self.parser.add_argument('--log-dir', default='runs/exp-0',
                                 help='path of data for save log.')
        self.parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                 help='number of epochs to train (default: 20)')
        self.parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                                 help='number of classify.')

    def parse(self):
        opt = self.parser.parse_args()

        return opt


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


def accuracy(output, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def confusion_matrix(y_true, y_pred, num_classes):
    """
    :param y_true:
    :param y_pred:
    :return:
    """

    CM = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    for i in range(len(y_true)):
        x = y_pred[i]
        y = y_true[i]
        if y >= num_classes or y < 0:
            continue
        CM[y][x] += 1

    return CM


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    m = epoch // 5
    lr = args.lr * (0.1 ** m)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_data(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='./data/', download=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='./data/', download=True, train=False, transform=transform),
        batch_size=args.test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def visualize(data_loader):
    """
    :param data_loader:
    :return:
    """
    import matplotlib.pyplot as plt

    for i_batch, (images_batch, label_batch) in enumerate(data_loader):
        print(i_batch, label_batch)

        if i_batch == 0:
            plt.figure()
            grid = utils.make_grid(images_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.axis('off')
            plt.ioff()
            plt.show()

            break


def train(args, model, data_loader, optimizer, epoch, criterion, writer, iter_count):
    """"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    embeddings, nlabels = [], []
    for batch_idx, (data, target) in enumerate(data_loader['train']):
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        features, outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), data.size(0))
        acc = accuracy(outputs[0], target)[0].item()
        # pred = torch.max(outputs.data, 1)
        top1.update(acc, data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        embeddings.append(features.data.numpy())
        nlabels.append(target.data.numpy())

        writer.add_scalar('train/loss', loss.item(), iter_count)
        if batch_idx % (args.log_interval * 5) == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, batch_idx, len(data_loader['train']), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

            writer.add_scalar('train/accuracy', acc, global_step=iter_count)
        iter_count += 1
    embeddings = np.concatenate(embeddings, 0)
    nlabels = np.concatenate(nlabels, 0)
    visual_feature_space(embeddings, nlabels, args.num_classes, epoch, './data/train/')
    test(args, model, data_loader['test'], epoch, criterion, writer, iter_count)


def test(args, model, data_loader, epoch, criterion, writer, iter_count):
    """"""
    model.eval()
    losses = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()

    embeddings, nlabels = [], []
    with torch.no_grad():
        end = time.time()
        CM = torch.zeros(args.num_classes, args.num_classes, dtype=torch.float32)
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = Variable(data), Variable(target)

            features, outputs = model(data)
            loss = criterion(outputs, target)

            embeddings.append(features.data.numpy())
            nlabels.append(target.data.numpy())

            # measure accuracy and record loss
            acc = accuracy(outputs[0], target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc[0].item(), data.size(0))

            # get the index of the max log-probability
            pred = outputs[0].max(1, keepdim=True)[1]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            CM += confusion_matrix(target, pred, args.num_classes)

        print('\nConfusion matrix :')
        print(CM)
        print('Precision:')
        print(CM / CM.sum(dim=0))
        print('Recall:')
        recall = CM.t() / CM.sum(dim=1)
        print(recall.t())

        print('Validate: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            batch_idx, len(data_loader), batch_time=batch_time, loss=losses, top1=top1))
        writer.add_scalar('test/loss', losses.val, iter_count)
        writer.add_scalar('test/accuracy', top1.val, iter_count)

        embeddings = np.concatenate(embeddings, 0)
        nlabels = np.concatenate(nlabels, 0)
        visual_feature_space(embeddings, nlabels, args.num_classes, epoch, './data/test/')


def main():
    """"""
    op = Options()
    args = op.parse()
    train_loader, test_loader = load_data(args)

    # visualize(train_loader)
    data_loader = {
        'train': train_loader,
        'test': test_loader
    }
    model = SphereFace20()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=True)

    criterion = AngularSoftmaxWithLoss()
    with SummaryWriter(args.log_dir) as writer:
        input_data = Variable(torch.rand(32, 1, 28, 28))
        writer.add_graph(model, (input_data, ))
        for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(args, optimizer, epoch - 1)
            iter_count = int(np.ceil(len(train_loader) / args.batch_size)) * (epoch - 1)
            train(args, model, data_loader, optimizer, epoch, criterion, writer, iter_count)


if __name__ == '__main__':
    main()
