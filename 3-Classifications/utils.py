import os
import sys
import errno
import numpy as np
import shutil
import os.path as osp
import matplotlib.pyplot as plt
import scipy.io as sio

import torch


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


def plot_features(dataset_name, features, labels, num_classes, epoch, prefix, save_dir):
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


class classification_map(object):

    def __init__(self, model, all_loader, use_gpu, dataset_dict, dataset_name, save_directory,
                 model_name, prefix="Maps", dpi=300):
        self.model = model
        self.allloader = all_loader
        self.use_gpu = use_gpu
        self.dataset_dict = dataset_dict
        self.dataset_name = dataset_name
        self.save_directory = save_directory
        self.prefix = prefix
        self.dpi = dpi
        self.model_name = model_name

    def generate_map(self):
        dataset_width = self.dataset_dict.dataset_width
        dataset_height = self.dataset_dict.dataset_height
        gt = self.dataset_dict.gt.flatten()
        total_indices = self.dataset_dict.total_indices
        self.model.eval()
        # Predict labels
        with torch.no_grad():
            y_hat = []
            for data, labels in self.allloader:
                if self.use_gpu:
                    data, labels = data.cuda(), labels.cuda()
                y_hat.extend(self.model(data).cpu().argmax(axis=1).detach().numpy())
            pred_labels = np.zeros(gt.shape)
            pred_labels[total_indices] = y_hat
            pred_labels = np.ravel(pred_labels)
            y_pred_list = self.list_color(pred_labels)
            y_pred_matrix = np.reshape(y_pred_list, (dataset_height, dataset_width, 3))
            # Generate Image
            fig = plt.figure(frameon=False)
            fig.set_size_inches(dataset_height * 2.0 / self.dpi, dataset_width * 2.0 / self.dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            fig.add_axes(ax)
            dirname = osp.join(self.save_directory, self.prefix)
            if not osp.exists(dirname):
                os.mkdir(dirname)
            save_name = osp.join(dirname, self.dataset_name + "_" + self.model_name + '.png')
            ax.imshow(y_pred_matrix)
            plt.savefig(save_name, dpi=self.dpi)
            plt.close()

    def list_color(self, y_hat):
        y = np.zeros((y_hat.shape[0], 3))
        if self.dataset_name == "SA":
            for index, item in enumerate(y_hat):
                if item == 0:
                    y[index] = np.array([140, 0, 191]) / 255.
                if item == 1:
                    y[index] = np.array([41, 177, 137]) / 255.
                if item == 2:
                    y[index] = np.array([0, 64, 255]) / 255.
                if item == 3:
                    y[index] = np.array([0, 128, 255]) / 255.
                if item == 4:
                    y[index] = np.array([0, 191, 255]) / 255.
                if item == 5:
                    y[index] = np.array([0, 255, 255]) / 255.
                if item == 6:
                    y[index] = np.array([64, 255, 191]) / 255.
                if item == 7:
                    y[index] = np.array([128, 255, 128]) / 255.
                if item == 8:
                    y[index] = np.array([191, 255, 64]) / 255.
                if item == 9:
                    y[index] = np.array([255, 255, 0]) / 255.
                if item == 10:
                    y[index] = np.array([255, 191, 0]) / 255.
                if item == 11:
                    y[index] = np.array([255, 128, 0]) / 255.
                if item == 12:
                    y[index] = np.array([255, 64, 0]) / 255.
                if item == 13:
                    y[index] = np.array([255, 0, 0]) / 255.
                if item == 14:
                    y[index] = np.array([191, 0, 0]) / 255.
                if item == 15:
                    y[index] = np.array([128, 0, 0]) / 255.
        if self.dataset_name == "KSC":
            for index, item in enumerate(y_hat):
                if item == 0:
                    y[index] = np.array([255, 128, 128]) / 255.
                if item == 1:
                    y[index] = np.array([255, 90, 1]) / 255.
                if item == 2:
                    y[index] = np.array([255, 2, 251]) / 255.
                if item == 3:
                    y[index] = np.array([193, 12, 190]) / 255.
                if item == 4:
                    y[index] = np.array([139, 68, 46]) / 255.
                if item == 5:
                    y[index] = np.array([172, 175, 84]) / 255.
                if item == 6:
                    y[index] = np.array([255, 220, 220]) / 255.
                if item == 7:
                    y[index] = np.array([145, 142, 142]) / 255.
                if item == 8:
                    y[index] = np.array([242, 240, 104]) / 255.
                if item == 9:
                    y[index] = np.array([255, 128, 81]) / 255.
                if item == 10:
                    y[index] = np.array([128, 128, 255]) / 255.
                if item == 11:
                    y[index] = np.array([71, 71, 9]) / 255.
                if item == 12:
                    y[index] = np.array([2, 177, 255]) / 255.
        if self.dataset_name == "UP":
            for index, item in enumerate(y_hat):
                if item == 0:
                    y[index] = np.array([255, 16, 35]) / 255.
                if item == 1:
                    y[index] = np.array([38, 214, 42]) / 255.
                if item == 2:
                    y[index] = np.array([14, 185, 228]) / 255.
                if item == 3:
                    y[index] = np.array([226, 218, 137]) / 255.
                if item == 4:
                    y[index] = np.array([203, 115, 206]) / 255.
                if item == 5:
                    y[index] = np.array([221, 168, 85]) / 255.
                if item == 6:
                    y[index] = np.array([142, 144, 87]) / 255.
                if item == 7:
                    y[index] = np.array([150, 120, 120]) / 255.
                if item == 8:
                    y[index] = np.array([51, 51, 153]) / 255.
        return y
