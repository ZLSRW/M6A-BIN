import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Utils import *
import pandas as pd
import csv
from .configure import *

class Cross_Structure_Perception(nn.Module):  # 对序列的进一步处理，因为是前向传播过程所以并不需要可逆
    def __init__(self, batch, unit, device="cuda:0"):
        super(Cross_Structure_Perception, self).__init__()
        self.batch = batch
        self.unit = unit

        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(self.unit)

        # 三维权重，使用哈达玛乘积；二维权重则使用矩阵乘法

        # self.weight_key=nn.Parameter(torch.zeros(size=(self.batch, self.unit, self.unit)))
        # self.weight_query=nn.Parameter(torch.zeros(size=(self.batch, self.unit, self.unit)))
        # self.weight_value=nn.Parameter(torch.zeros(size=(self.batch, self.unit, self.unit)))

        # 假设两个图之间信息感知的程度是一致的
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        self.weight_value = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))

        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_value.data, gain=1.414)
        self.to(device)

    def Perception_coefficients(self, A):
        # bat, N, fea = A1.size() #32 140 140
        # print(type(A1))
        # print(type(A2))
        # print(A1.shape)
        # print(A2.shape)
        # print(type(A1))
        # print(A1)
        key = torch.matmul(A, self.weight_key)
        query = torch.matmul(A, self.weight_query)
        value = torch.matmul(A, self.weight_value)

        data = torch.matmul(query, key.permute(0, 2, 1))
        data = self.bn(data)
        data = self.relu(data) #暂时去掉
        coefficient = F.softmax(data, dim=2)
        Perception = coefficient @ A @ value
        # print('Perception[0] '+str(Perception[0]))

        # 标准化、mask、对称归一化 (去掉)
        Perception = normalize_and_symmetrize_tensor(Perception, 0.5)
        # print('Perception[0] '+str(Perception[0]))

        return Perception

    def forward(self, A):  # 希望A1,A2分别为对称归一化矩阵
        # print(type(A))
        Perception = self.Perception_coefficients(A)
        A = 0.5 * (Perception+Perception.permute(0,2,1))  # 严格保持对称归一化
        return A


class GraphCouplingLayer(nn.Module):
    def __init__(self, batch, size, device="cuda:0"):
        super(GraphCouplingLayer, self).__init__()
        self.batch = batch
        self.size = size
        self.informationTrans = Cross_Structure_Perception(self.batch,self.size)

    def forward(self, A1, A2, invertible):
        if not invertible:
            B1 = A1
            B2 = A2 + self.informationTrans(A1)
            return torch.tensor(B1), torch.tensor(B2)

        B1 = A1
        B2 = A2 - self.informationTrans(B1)
        return torch.tensor(B1), torch.tensor(B2)


class Graph_Interaction(nn.Module):
    def __init__(self, num, batch,size,device="cuda:0"):
        super(Graph_Interaction, self).__init__()
        self.batch=batch
        self.size=size
        self.coupling = GraphCouplingLayer(self.batch,self.size)
        self.Graph_Coupling_Layers = nn.ModuleList()
        for _ in range(num + 1):
            self.Graph_Coupling_Layers.append(self.coupling)

    def forward(self, A1, A2, invertible):  # 警告要添加全局声明，可能会报错
        # global B1, B2
        # print('A1[1] '+str(A1[0]))
        # print('A2[1] '+str(A2[0]))
        for i, layer in enumerate(self.Graph_Coupling_Layers):
            if i % 2 == 0:
                B1, B2 = layer(A1, A2, invertible)
            else:
                B1, B2 = layer(A2, A1, invertible)
            # print('调用')
            A1 = B1
            A2 = B2
        return B1, B2
