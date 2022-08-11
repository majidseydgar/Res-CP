import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import PReLU
# from torchsummary import summary


class Residual(nn.Module):  # pytorch
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 use_1x1conv=False,
                 stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride), PReLU())
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)
        self.p_relu = nn.Sequential(PReLU())
        if use_1x1conv:
            self.conv3 = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, X):
        Y = self.p_relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return self.p_relu(Y + X)


class RCN(nn.Module):
    def __init__(self, band, num_classes):
        super(RCN, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=24,
            kernel_size=(1, 1, 7),
            stride=(1, 1, 2))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24),  # 0.1
            PReLU())

        self.res_net1 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net2 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(24, 24, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(24, 24, (3, 3, 1), (1, 1, 0))

        kernel_3d = math.ceil((band - 6) / 2)

        self.conv2 = nn.Conv3d(
            in_channels=24,
            out_channels=128,
            padding=(0, 0, 0),
            kernel_size=(1, 1, kernel_3d),
            stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128),  # 0.1
            PReLU())
        self.conv3 = nn.Conv3d(
            in_channels=1,
            out_channels=24,
            padding=(0, 0, 0),
            kernel_size=(3, 3, 128),
            stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(24),  # 0.1
            PReLU())

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.center_fc = nn.Sequential(nn.Linear(24, 2))
        self.full_connection = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2, num_classes)  # ,
            # nn.Softmax()
        )

    def forward(self, X):
        x1 = self.batch_norm1(self.conv1(X))
        # print('x1', x1.shape)

        x2 = self.res_net1(x1)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        x5 = self.center_fc(x4)
        return x4, x5, self.full_connection(x5)


__factory = {"RCN": RCN, }


def create(name, band, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](band, num_classes)


if __name__ == '__main__':
    pass

