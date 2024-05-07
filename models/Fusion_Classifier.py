'''''
主要完成三个函数：
（1）基于节点级表示（两个bx41x64的矩阵）获得图级表示（两个bx64的矩阵）
（2）对两个图级表示进行融合（对两个bx64的矩阵进行融合）。
（3）进行分类（对融合后的）
'''''
# 节点级表示进行融合
import torch
from torch import nn
from .fuse import *
from .configure import *

# 图表示池化
class GraphRepresentation(nn.Module): #这部分可能会报错
    def __init__(self, inputsize, outputsize, device):
        super(GraphRepresentation, self).__init__()
        self.tanh = nn.Tanh()
        self.device = device
        self.inputsize = inputsize
        self.outputsize = outputsize
        # 加个标准化层
        # self.bn1 = nn.BatchNorm1d(256)
        # self.bn2 = nn.BatchNorm1d(256)

        self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(256)

        # self.expansion= nn.Sequential(
        #     nn.Linear(768,256),
        #     nn.ReLU(), #权重
        #     nn.Linear(256, 256)
        # )

        # Elom特征的融合，特征维度为41x768
        self.expansion = nn.Sequential(
            nn.Linear(256, 128),
            # nn.Tanh(),
            nn.ReLU(),  # 权重
            nn.Linear(128, 64)
        )
        self.prob = nn.Sequential(
            nn.Linear(64, 64),
            nn.Softmax(),  # 权重
            # nn.Sigmoid(),  # 权重
            nn.Linear(64, 1),
        )

        self.prob2 = nn.Sequential(
            nn.Linear(41, 41),
            nn.Softmax(),  # 权重
            # nn.Sigmoid(),  # 权重
        )


        self.fc_shape1 = nn.Sequential(
            nn.Linear(41, 41),
            # nn.LeakyReLU(),
            # nn.Tanh(), #效果不错
            nn.ReLU(),
            nn.Linear(41, 1),
        )

        # self.expansion = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU(),  # 权重
        #     nn.Linear(128, 64)
        # )

        # self.prob = nn.Sequential( #这个prob存在一定的问题、
        #     nn.Linear(256, 128),
        #     nn.Softmax(),  # 权重
        #     # nn.Sigmoid(),  # 权重
        #     nn.Linear(128, 1),
        # )

        self.pool = nn.Sequential(
            nn.Linear(41, 41),
            # nn.ReLU(),  # 权重
            nn.Sigmoid(),  # 权重
            nn.Linear(41, 1),
        )

        # # 只传递标签
        # self.expansion= nn.Sequential(
        #     nn.Linear(4,64),
        #     nn.ReLU(), #权重
        #     nn.Linear(64, 64)
        # )
        # self.prob = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.Softmax(),  # 权重
        #     nn.Linear(64, 1),
        # )

        self.to(device)

    def forward(self, R): #加入残差+标准化，考虑是否降低
        # print(R.shape)
        # 全连接方式
        # R_s=self.fc_shape1(R.permute(0,2,1)).permute(0,2,1)
        # R_s = self.bn1(R_s.permute(0, 2, 1)).permute(0, 2, 1).squeeze()
        # R_prob=self.prob(R)

        # 加权方式
        R = self.expansion(R)
        # R_prob=self.prob(R)
        # R_prob = self.prob2(self.prob(R).permute(0,2,1)).permute(0,2,1)
        R_prob = self.prob2(self.prob(R).permute(0,2,1)).permute(0,2,1)
        R_s = torch.sum(R_prob * R, dim=-2)  # 706x41
        # print(R_s.shape)
        R_s =self.bn1(R_s)

        # Elom4_R_s=self.fc_shape1(R.permute(0,2,1)).permute(0,2,1)
        # Elom4_R_s = self.bn2(Elom4_R_s.permute(0, 2, 1)).permute(0, 2, 1).squeeze() #batsize*256
        # Elom4_R_prob=self.prob(Elom4_R)

        # print(Elom4_R.shape)
        # print(R.shape)

        # Representation=self.bn1(R.permute(1,0)).permute(1,0)+self.bn2(Elom4_R.permute(1,0)).permute(1,0)
        # Representation=Elom4_R_s+R_s
        Representation=R_s

        # R=self.pool(R.permute(0,2,1)).squeeze()
        # R=self.bn(R)
        return Representation, R_prob


# 图表示融合
class GraphRepresentationfusion(nn.Module):
    def __init__(self, device):
        super(GraphRepresentationfusion, self).__init__()
        self.softmax = nn.Softmax()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fushion=fuse(64,64)

        self.to(device)

    def forward(self, R1, R2):
        # print(R1.shape)
        # 暂时只是简单加权
        # R=self.bn1(R1.permute(0,2,1)).permute(0,2,1)+self.bn2(R2.permute(0,2,1)).permute(0,2,1)
        # R=self.bn1(R1)+self.bn1(R2) # 感觉还行
        # R=self.bn1(R1+R2)
        # R = R1 + R2
        R = self.fushion(R1,R2)
        # R=0.5*(R1+R2)
        # R = self.bn1(R)
        return R


# 分类
class Classifier(nn.Module):
    def __init__(self, inputsize, outputsize, device):
        super(Classifier, self).__init__()
        self.inputsize = 64
        self.outputsize = 64
        self.bn=nn.BatchNorm1d(256)
        self.classifier = nn.Sequential(
            nn.Linear(self.inputsize, 1),
            nn.Sigmoid(),
        )
        self.to(device)

    def forward(self, R):
        return self.classifier(R)
