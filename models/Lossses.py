"""""
三个部分的包装和目的：（1）图的可逆块：目的是通过逆过程中的置信度池化将网络中重要的拓扑结构提取出来。（2）图、表示的空间对齐（拉普拉斯矩阵是对称的，因此只需要考虑只求一半）：将非欧的结构性质通过前后表征进行还原对齐。（3）孪生耦合层：m函数对表征进一步处理，以耦合的方式对学习的表征进一步增强。
孪生部分是为了从表征的角度对图进行约束。
计算相关损失，包括：（1）图可逆部分的损失。（2）加性耦合部分的损失。（3）孪生部分的度量损失。最理想的状态是将这三部分的损失进行融合，并添加一定正则。
"""""
import torch
import numpy as np
from .Utils import *
from torch import nn
from .configure import *

# 图池化损失（图级交互的损失也包含在其中，不能直接在handler进行计算，应该在哈达玛乘积更新之后）
def graph_neg_log_likelihood(output, model):
    """compute the log likelihood with change of variables formula, average per pixel"""
    N, D, _ = output.shape  # batch size and single output size

    """First summand"""
    constant = torch.from_numpy(np.array(0.5 * D * D * N * np.log(np.pi))).type(torch.float64)

    """Second summand"""
    sum_squared_mappings = torch.square(output)
    sum_squared_mappings = torch.sum(sum_squared_mappings)
    sum_squared_mappings = 0.5 * sum_squared_mappings

    """行列式计算"""
    """log diagonals of U"""
    log_diagonals_triu = []
    for param in model.parameters():
        # print(param.shape) #调用两次，一次为上三角一次为下三角,706x41x41
        # if len(param.shape) == 3 and param[0, 0, 0] != 0:  # if upper triangular and matrix
            # log_diagonals_triu.append(torch.log(torch.abs(torch.diag(param)))) #权重为三维张量，需要重写该方法
        if param[0, 0, 0] != 0:  # Check if param is a 3D tensor and has square matrices
            log_diagonals_triu = []
            for i in range(param.size(0)):  # Loop through each 2D matrix
                diag_elements = torch.diagonal(param[i])  # Extract diagonal elements
                log_diagonals_triu.append(torch.log(torch.abs(diag_elements)))

    """lu-block M"""
    last = log_diagonals_triu[len(log_diagonals_triu) - 1]
    last = torch.sum(last)

    output = constant + sum_squared_mappings - last

    return output/(N*D*D)

# 孪生损失 (在表示可逆块前计算)
import torch



def wasserstein_distance_sum_average(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape."

    batch_size, feature_num, feature_dim = tensor1.shape
    device = torch.device('cuda:0')
    wasserstein_sum = 0.0
    for i in range(batch_size):
        # print(i)
        # 取出两个张量对应批次的特征
        for j in range(feature_num):
            C = tensor1[i][j]
            D = tensor2[i][j]

            # C_normalized = (C - C.min()) / (
            #             C.max()- C.min())
            # D_normalized = (D - D.min()) / (
            #             D.max() - D.min())

            # 计算距离矩阵M（使用欧氏距离）
            # M = torch.norm(C_normalized - D_normalized)
            M = torch.norm(C - D)
            # print(M)

            # 将计算的沃瑟斯坦距离累加到总和中
            wasserstein_sum += M

    # 求关于批次的平均沃瑟斯坦距离
    wasserstein_avg = wasserstein_sum / (batch_size*feature_num*feature_dim) #将元素woza
    # print('wasserstein_distance')
    return torch.abs(torch.tensor(wasserstein_avg).to(device))

# bn1=nn.BatchNorm1d(256)
# 表示级交互的可逆损失（不增加方差的版本）
def representation_neg_log_likelihood(output, model):
    # a_coff=1e-5
    a_coff=1
    """compute the log likelihood with change of variables formula, average per pixel"""
    N, D, f = output.shape  # batch size and single output size
    # output=bn1(output.permute(0,2,1)).permute(0,2,1)

    """First"""
    constant = torch.from_numpy(np.array(0.5 * D * f * N * np.log(np.pi))).type(torch.float64)

    """Second"""
    # print(output.shape) # 706x41x256
    # 进行归一化
    sum_squared_mappings = torch.square(output)
    sum_squared_mappings = torch.sum(torch.mean(sum_squared_mappings,dim=2))
    sum_squared_mappings = 0.5 * sum_squared_mappings

    # # 试试
    # sum_squared_mappings = torch.square(output)
    # sum_squared_mappings = torch.sum(sum_squared_mappings)
    # sum_squared_mappings = 0.5 * sum_squared_mappings

    # """方差log"""
    # log_diagonals_triu = []
    # for param in model.parameters():
    #     if len(param.shape) == 2 and param[1, 0] == 0:  # if upper triangular and matrix
    #         log_diagonals_triu.append(torch.log(torch.abs(torch.diag(param)))) #只有上三角矩阵有值，下三角的对角线元素为0
    #
    # """lu-block M"""
    # last = log_diagonals_triu[len(log_diagonals_triu) - 1]
    # last = torch.sum(last)
    # 雅可比行列式直接为0
    last=0

    output = constant + sum_squared_mappings - last
    # print('output '+str(output))
    # print('sum_squared_mappings '+str(sum_squared_mappings))
    # print('output/(N*D*f) '+str(output/(N*D*f)))
    # print('(N*D*f) '+str(N*D*f))

    return output/(N*D*f*256*4*5)