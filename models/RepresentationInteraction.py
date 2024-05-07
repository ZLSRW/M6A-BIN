import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Utils import *
import pandas as pd
import csv
from torch.nn.utils import weight_norm
from .configure import *

class mfuse(nn.Module):  # 时序块
    def __init__(self, n_inputs=41, n_outputs=41, kernel_size=1, stride=1, padding=0, dropout=0.2):  # 需要仔细看看取值
        super(mfuse, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           padding=padding, stride=stride))  # 权重归一化
        self.relu1 = nn.ReLU()
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)  # 一维卷积1

        # self.bn1 = nn.BatchNorm1d(n_outputs)
        # self.bn2 = nn.BatchNorm1d(n_outputs)
        self.bn1 = nn.BatchNorm1d(256)
        # self.bn2 = nn.BatchNorm1d(768)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_inputs, kernel_size,
                                           padding=padding, stride=stride))
        self.relu2 = nn.ReLU()
        self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)  # 一维卷积2

        self.leakyReLU1 = nn.LeakyReLU()
        self.leakyReLU2 = nn.LeakyReLU()

        self.net = nn.Sequential(self.conv1, self.relu1,
                                 self.bn1,
                                 # self.conv2, self.relu2,
                                 # self.bn2
                                 )
        self.net1 = nn.Sequential(self.conv1, self.relu1)
        self.net2 = nn.Sequential(self.conv2, self.relu2)

        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net1(x)
        out = self.bn1(out.permute(0, 2, 1)).permute(0, 2, 1)
        # out = self.net2(out)
        # out = self.bn2(out.permute(0, 2, 1)).permute(0, 2, 1)
        #
        # out = self.net(x)

        # # 系数计算（1）只使用softmax，
        coff = self.softmax(out)

        # # （2）使用softmax+sigmoid,不加全连接层不知道会不会出问题,效果不佳
        # cf1=self.softmax(out)
        # cf2=self.sigmoid(out)
        # coff=self.sigmoid(cf1*cf2)

        out = coff * out

        return self.tanh1(out)
        # return out #效果不如tanh


# class CouplingLayer(nn.Module):
#     def __init__(self, device="cuda:0"):
#         super(CouplingLayer, self).__init__()
#         self.m = mfuse()
#
#     def forward(self, x1, x2, invertible):
#         if not invertible:
#             y1 = x1
#             y2 = x2 + self.m(x1)
#             return y1, y2
#
#         y1 = x1
#         y2 = x2 - self.m(y1)
#         return y1, y2

class CouplingLayer(nn.Module):
    def __init__(self, device="cuda:0"):
        super(CouplingLayer, self).__init__()
        self.m = mfuse()
        self.m1 = mfuse()
        self.m2 = mfuse()

    def forward(self, x1, x2, invertible):
        if not invertible:
            # y1 = x1
            # y2 = x2 - self.m(x1)
            #
            # y3 = y1 + self.m(y2)
            # y4 = y2
            y1 = x1
            y2 = x2 - self.m(x1)

            y3 = y1 - self.m(y2)
            y4 = y2

            return y3, y4

        y1 = x1
        y2 = x2 + self.m(y1)

        y3 = y2
        y4 = y1 + self.m(y3)
        return y3, y4


class Siamese_Coupling(nn.Module):
    def __init__(self, num, device="cuda:0"):
        super(Siamese_Coupling, self).__init__()
        self.coupling = CouplingLayer()
        # self.coupling1 = CouplingLayer()
        # self.coupling2 = CouplingLayer()
        self.Siamese_Coupling_Layer = nn.ModuleList()
        for _ in range(num):  # 先试试一层
            self.Siamese_Coupling_Layer.append(self.coupling)
        # self.Siamese_Coupling_Layer.append(self.coupling1)
        # self.Siamese_Coupling_Layer.append(self.coupling2)
        # self.bn1 = nn.BatchNorm1d(768)
        # self.bn2 = nn.BatchNorm1d(768)
        # self.bn1 = nn.BatchNorm1d(41)
        # self.bn2 = nn.BatchNorm1d(41)

    def forward(self, x1, x2, invertible):
        # x1=self.bn1(x1.permute(0,2,1)).permute(0,2,1)
        # x2=self.bn2(x2.permute(0,2,1)).permute(0,2,1)
        # x1=self.bn1(x1)
        # x2=self.bn2(x2)
        for i, layer in enumerate(self.Siamese_Coupling_Layer):
            if i % 2 == 0:
                y1, y2 = layer(x1, x2, invertible)
            else:
                y1, y2 = layer(x2, x1, invertible)
            x1 = y1
            x2 = y2
        return y1, y2
