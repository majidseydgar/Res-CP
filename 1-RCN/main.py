import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from utils import encoded_features, plot_features
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import datasets
import models
from utils import AverageMeter, Logger
from center_loss import CenterLoss

matplotlib.use('Agg')

parser = argparse.ArgumentParser("Center Loss Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='SA', choices=['UP', 'SA', 'KSC', 'UH'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('-dir', '--dataset-dir', default="Datasets/", type=str,
                    help="Directory to the dataset")
parser.add_argument("--patch-len", type=int, default=4,
                    help="the length of patch extraction, e.g., default is (4*2)+1 = 9")
parser.add_argument("--sample-num", type=int, default=10,
                    help="Number of training samples in each class")
# optimization
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--lr-model', type=float, default=0.003, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=500)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--etta', type=float, default=0)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='RCN')
# misc
parser.add_argument('--eval-freq', type=int, default=500)
parser.add_argument('--print-freq', type=int, default=32)
parser.add_argument('--features_num', type=int, default=24)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1334)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log/RCN')
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")
parser.add_argument('--emb', action='store_true', help="if you want to save the embedding features")

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '_.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))
    if args.emb:
        dataset = datasets.Dataset_call(dataset_name=args.dataset, root=args.dataset_dir,
                                        patch_length=args.patch_len, return_zero_labels=True,
                                        sample_num=args.sample_num)
        trainloader, testloader, allloader = dataset.get_iterator()
    else:
        dataset = datasets.Dataset_call(dataset_name=args.dataset, root=args.dataset_dir,
                                        patch_length=args.patch_len, return_zero_labels=False,
                                        sample_num=args.sample_num)
        trainloader, testloader = dataset.get_iterator()

    print("Creating model: {}".format(args.model))
    model = models.create(name=args.model, num_classes=dataset.num_classes, band=dataset.Input_dimension)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=dataset.num_classes, feat_dim=2, use_gpu=use_gpu)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    # optimizer_model = torch.optim.RMSprop(model.parameters(), lr=args.lr_model, eps=1e-8, weight_decay=0)
    optimizer_centloss = torch.optim.RMSprop(criterion_cent.parameters(), lr=args.lr_cent)

    if args.stepsize > 0:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer_model,
                                                               args.stepsize, eta_min=args.etta, last_epoch=-1)
        # scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()

    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        train(model, criterion_xent, criterion_cent,
              optimizer_model, optimizer_centloss,
              trainloader, use_gpu, dataset.num_classes, epoch)

        if args.stepsize > 0: scheduler.step()

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            acc, err = test(model, testloader, use_gpu, dataset.num_classes, epoch)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

        if args.emb:
            if epoch + 1 == int(args.max_epoch):
                encoded_features(model=model, allloader=allloader, use_gpu=use_gpu, dataset_dict=dataset,
                                 dataset_name=args.dataset, save_dir=args.save_dir, prefix="embeddings",)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, criterion_xent, criterion_cent,
          optimizer_model, optimizer_centloss,
          trainloader, use_gpu, num_classes, epoch):
    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()

    if args.plot:
        all_features, all_labels = [], []

    for batch_idx, (data, labels) in enumerate(tqdm(trainloader)):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        _, features, outputs = model(data)
        # print("Output shape", outputs.shape, "features shape", features.shape, "label shape", labels.shape)
        loss_xent = criterion_xent(outputs, labels.long())
        loss_cent = criterion_cent(features, labels)
        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / args.weight_cent)
        optimizer_centloss.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))

        if args.plot:
            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(labels.data.numpy())

        if (batch_idx + 1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})" \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg,
                          cent_losses.val, cent_losses.avg))

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(dataset_name=args.dataset, features=all_features, labels=all_labels,
                      num_classes=num_classes, epoch=epoch, save_dir=args.save_dir,
                      prefix='train', )


def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            _, features, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            if args.plot:
                if use_gpu:
                    all_features.append(features.data.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())
                else:
                    all_features.append(features.data.numpy())
                    all_labels.append(labels.data.numpy())

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(dataset_name=args.dataset, features=all_features, labels=all_labels,
                      num_classes=num_classes, epoch=epoch, save_dir=args.save_dir,
                      prefix='test', )

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err


if __name__ == '__main__':
    main()
