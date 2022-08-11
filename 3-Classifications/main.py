import collections
import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
import scipy.io as sio
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from operator import truediv
from sklearn import metrics
from utils import classification_map
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import datasets
import models
from utils import AverageMeter, Logger

matplotlib.use('Agg')

parser = argparse.ArgumentParser("Center Loss Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='UP', choices=['UP', 'SA', 'KSC', 'UH'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('-dir', '--dataset-dir', default="Datasets/", type=str,
                    help="Directory to the dataset")
parser.add_argument("--patch-len", type=int, default=4,
                    help="the length of patch extraction, e.g., default is (4*2)+1 = 9")
parser.add_argument("--val-size", type=int, default=0.3,
                    help="the ratio of training samples reserved for validation")
# optimization
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--lr-model', type=float, default=0.1e-3, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='SSRN', choices=['SSRN'])
# misc
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=200)
parser.add_argument('--features_num', type=int, default=24)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1334)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log/Classification')
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")
parser.add_argument('--generate-png', action='store_true',
                    help="if you want to generate the classification map")

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + "_" + args.model + '_.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))
    if args.generate_png:
        dataset = datasets.Dataset_call(dataset_name=args.dataset, root=args.dataset_dir,
                                        patch_length=args.patch_len, return_zero_labels=True,
                                        sample_num=args.val_size)
        trainloader, val_loader, testloader, allloader = dataset.get_iterator()
    else:
        dataset = datasets.Dataset_call(dataset_name=args.dataset, root=args.dataset_dir,
                                        patch_length=args.patch_len, sample_num=args.val_size)
        trainloader, val_loader, testloader = dataset.get_iterator()

    print("Creating model: {}".format(args.model))
    model = models.create(name=args.model, num_classes=dataset.num_classes, band=dataset.Input_dimension)
    # model = models.SSRN_network(band=dataset.Input_dimension, num_classes=dataset.num_classes)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    criterion_xent = nn.CrossEntropyLoss()
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_model, betas=(0.9, 0.999), eps=1e-8,
                                       weight_decay=0)
    # optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer_model,
        #                                                        args.stepsize, eta_min=args.etta, last_epoch=-1)
    start_time = time.time()

    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        train(model, criterion_xent, optimizer_model,
              trainloader, use_gpu, dataset.num_classes, epoch)

        if args.stepsize > 0: scheduler.step()

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 and epoch + 1 != args.max_epoch:
            print("==> Validation")
            acc, err = test(model=model, testloader=val_loader, use_gpu=use_gpu, loss_x=criterion_xent, epoch=epoch)
            print("Validation Accuracy (%): {}\t Validation Error rate (%): {}".format(acc, err))

        elif (epoch + 1) == args.max_epoch:
            print("==> Test")
            Overall_Accuracy, Average_accuracy, kappa = test(model=model, testloader=testloader, use_gpu=use_gpu,
                                                             loss_x=criterion_xent, epoch=epoch)
            print("Overall Accuracy (%): {}\n Kappa Coefficient (%): {}"
                  "\n Average Accuracy (%): {}".format(Overall_Accuracy, kappa, Average_accuracy))

        if args.generate_png:
            if epoch + 1 == int(args.max_epoch):
                classification_map(model=model, all_loader=allloader, use_gpu=use_gpu,
                                   dataset_dict=dataset, model_name=args.model,
                                   dataset_name=args.dataset, save_directory=args.save_dir,).generate_map()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, criterion_xent,
          optimizer_model, trainloader, use_gpu, num_classes, epoch):
    model.train()
    xent_losses = AverageMeter()
    losses = AverageMeter()

    for batch_idx, (data, labels) in enumerate(tqdm(trainloader)):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs = model(data)
        loss_xent = criterion_xent(outputs, labels.long())
        loss = loss_xent
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))


def test(model, testloader, use_gpu, loss_x, epoch, prefix="trained model"):
    model.eval()
    correct, total = 0, 0
    best_valid_loss = float('inf')
    losses = AverageMeter()
    xent_losses = AverageMeter()

    dirname = osp.join(args.save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, args.dataset)

    if epoch + 1 == args.max_epoch:
        y_hat = []
        y_true = []
        model.load_state_dict(torch.load(save_name + "-" + args.model + '-model.pt'))
        with torch.no_grad():
            for data, labels in testloader:
                if use_gpu:
                    data, labels = data.cuda(), labels.cuda()
                y_hat.extend(np.array(model(data).cpu().argmax(axis=1)))
                y_true.extend(np.array(labels.cpu()))
        collections.Counter(y_hat)
        collections.Counter(y_true)

        Overall_Accuracy = metrics.accuracy_score(y_hat, y_true)
        confusion_matrix = metrics.confusion_matrix(y_hat, y_true)
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        np.seterr(divide='ignore', invalid='ignore')
        Accuracy_per_Class = np.nan_to_num(truediv(list_diag, list_raw_sum))
        Average_accuracy = np.mean(Accuracy_per_Class)
        kappa = metrics.cohen_kappa_score(y_hat, y_true)
        return Overall_Accuracy * 100, Average_accuracy * 100, kappa * 100

    else:
        with torch.no_grad():
            for data, labels in testloader:
                if use_gpu:
                    data, labels = data.cuda(), labels.cuda()
                outputs = model(data)
                if epoch + 1 < args.max_epoch:
                    loss_xent = loss_x(outputs, labels.long())
                    loss = loss_xent
                    if loss_xent < best_valid_loss:
                        best_valid_loss = loss_xent
                        torch.save(model.state_dict(), save_name + "-" + args.model + '-model.pt')
                    losses.update(loss.item(), labels.size(0))
                    xent_losses.update(loss_xent.item(), labels.size(0))
                correct += (outputs.argmax(dim=1) == labels).sum().cpu().item()
                total += labels.size(0)
                # correct += (predictions == labels.data).sum()

        acc = correct * 100. / total
        err = 100. - acc
        if epoch + 1 < args.max_epoch:
            print("Epoch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(epoch + 1, args.max_epoch, losses.val, losses.avg, ))
        return acc, err


if __name__ == '__main__':
    main()
