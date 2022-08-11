import os
import sys
import errno
import shutil
import matplotlib.pyplot as plt
import os.path as osp
import torch
import numpy as np
import scipy.io as sio


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def plot_features(dataset_name, features, labels, num_classes, epoch, save_dir, prefix):
    """Plot features on 2D plane.
    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    if dataset_name == "UP":
        colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for label_idx in range(num_classes):
            plt.scatter(
                features[labels == label_idx, 0],
                features[labels == label_idx, 1],
                c=colors[label_idx],
                s=1,
            )
        plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')

    if dataset_name == "SA":
        colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                  'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16']
        for label_idx in range(num_classes):
            plt.scatter(
                features[labels == label_idx, 0],
                features[labels == label_idx, 1],
                c=colors[label_idx],
                s=1,
            )
        plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9',
                    '10', '11', '12', '13', '14', '15', '16'], loc='upper right')

    if dataset_name == "KSC":
        colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                  'C10', 'C11', 'C12', 'C13']
        for label_idx in range(num_classes):
            plt.scatter(
                features[labels == label_idx, 0],
                features[labels == label_idx, 1],
                c=colors[label_idx],
                s=1,
            )
        plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9',
                    '10', '11', '12', '13'], loc='upper right')

    dirname = osp.join(save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch + 1) + dataset_name + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


def encoded_features(model, allloader, use_gpu, dataset_dict, save_dir,
                     dataset_name, prefix="embeddings"):
    dataset_width = dataset_dict.dataset_width
    dataset_height = dataset_dict.dataset_height
    model.eval()
    with torch.no_grad():
        embeddings = []
        for data, labels in allloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            pred, _, _ = model(data)
            embeddings.extend(pred.cpu().detach().numpy())
            # embeddings.extend(pred.float().cpu().detach().numpy())

        embeddings = np.array(embeddings)
        dirname = osp.join(save_dir, prefix)
        if not osp.exists(dirname):
            os.mkdir(dirname)
        save_name = osp.join(dirname, dataset_name)
        embeddings = embeddings.reshape((dataset_height, dataset_width, -1))
        sio.savemat(save_name + "_embeddings", {'embeddings' + dataset_name: embeddings})
        train_set = np.zeros(dataset_width * dataset_height)
        train_set[dataset_dict.train_indices] = dataset_dict.y_train + 1
        sio.savemat(save_name + "_train_set_idx", {'train_set_idx_' + dataset_name:
                                                       train_set.reshape((dataset_height, dataset_width)).astype(int)})
        test_set = np.zeros(dataset_width * dataset_height)
        test_set[dataset_dict.test_indices] = dataset_dict.y_test + 1
        sio.savemat(save_name + "_test_set_idx", {'test_set_idx_' + dataset_name:
                                                      test_set.reshape((dataset_height, dataset_width)).astype(int)})
