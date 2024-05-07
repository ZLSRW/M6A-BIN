"""""
(1) 输入序列的两种表示（初始化表示、序列二级结构图），构建序列的两种图,以及序列的初始表示。在生物序列中，则一个为原始序列对应的图，一个为二级结构序列对应的图。在其他领域，可以类似对比学习，将两种图认为是两种增广形式。（在另一个文件完成，需要满足对称，并尽可能实现归一化）
(2) 设计两个权重矩阵，以哈达玛乘积的形式对生成的图进行池化。（Confidence Pooling）。权重矩阵需要保留，一方面用于motif或pattern的提取（保留高权重），另一方面用于图卷积的可逆，进行图的还原。
(3) 基于图获得两种拉普拉斯算子（D-W），不进行归一化，归一化操作尝试在图化的环节完成。
(4) 图结构和序列表示的对齐（或者称为双图对齐，切比雪夫近似，使用三阶）。接收一种序列表示，作为节点表示（one-hot或cgr或transformer），进行谱图卷积。+可逆
(5) 表示的孪生耦合。+可逆

部分类缺乏super函数，可能会报错
"""""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Utils import *
from .RepresentationInteraction import *
import pandas as pd
import csv
from .GraphInteraction import *
from .Lossses import *
from .configure import *

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return


def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return


def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

# class 1: 拉普拉斯矩阵生成（池化+（D-W））+可逆
# class 2: 空间对齐（切比雪夫近似） +可逆
# class 3: 孪生-耦合 +可逆

class PoolingLayer(nn.Module):
    # def __init__(self, size, batch, device="cuda:0"):
    def __init__(self, size, device="cuda:0"):
        super(PoolingLayer, self).__init__()
        self.size = size
        # self.batch = batch
        # self.weight = nn.Parameter(torch.rand(size=(self.batch, self.size, self.size)))
        self.weight = nn.Parameter(torch.rand(size=(1, self.size, self.size)))
        self.to(device)

        return

    def forward(self, A):
        A = torch.mul(self.weight, A)
        A =0.5*(A+A.permute(0,2,1)) #可逆部分需要写出这部分的计算
        # print(self.weight)
        return A

class TransposeHook: #保证在更新梯度时，U和L的权重互为转置，这样的操作可能会影响模型的训练和收敛性
        def __init__(self, source_layer):
            self.source_layer = source_layer

        def __call__(self, grad):
            self.source_layer.weight.data.copy_(self.source_layer.weight.data.permute(0, 2, 1))
            return grad

class LaplacianGeneration(nn.Module):
    def __init__(self,num, batch, size, fold, device="cuda:0"):
        super(LaplacianGeneration, self).__init__()

        self.num=num
        self.batch = batch
        self.size = size
        self.fold = fold

        self.graph_interaction = Graph_Interaction(num, batch, size)

        # self.U1 = PoolingLayer(size, batch)
        # self.U2 = PoolingLayer(size, batch)
        self.U1 = PoolingLayer(size)
        self.U2 = PoolingLayer(size)
        # self.L1 = PoolingLayer(size, batch)
        # self.L2 = PoolingLayer(size, batch)

        # 权重为三维矩阵，bx41x41
        # 定义两个全1的邻接矩阵，类似LU分解，分别计算其上三角或下三角矩阵。特别的，假设上三角矩阵为下三角矩阵的转置
        # 增加一部分内容，即L和U除了对角线，其余满足对称关系
        self.mask_triu = torch.triu(torch.ones(1, self.size, self.size)).bool()  # 返回上三角矩阵
        self.mask_tril = torch.tril(torch.ones(1, self.size, self.size)).bool()  # 返回下三角矩阵

        for i in range(1):
            self.mask_tril[i].fill_diagonal_(0)

        # 一类图池化
        with torch.no_grad():  # 反向传播时不自动求导
            self.U1.weight.copy_(torch.triu(self.U1.weight))
            self.U1.weight[i].fill_diagonal_(1)
        self.U1.weight.register_hook(get_zero_grad_hook(self.mask_triu, device)) #对梯度的控制
        # with torch.no_grad():
        #     self.L1.weight.copy_(self.U1.weight.permute(0,2,1)) #使得L层和U层的权重一致
        #     for i in range(self.batch):
        #         self.L1.weight[i].fill_diagonal_(1)
        #     self.L1.weight.copy_(self.L1.weight)
        # self.L1.weight.register_hook(get_zero_grad_hook(self.mask_tril, device))
        # self.L1.weight.register_hook(TransposeHook(self.U1))

        self.poolinglayer1 = nn.ModuleList()
        self.poolinglayer1.append(self.U1)
        # self.poolinglayer1.append(self.L1)

        # 二类图池化
        with torch.no_grad():  # 反向传播时不自动求导，节省内存。
            self.U2.weight.copy_(torch.triu(self.U2.weight))
            self.U2.weight[i].fill_diagonal_(1)
        self.U2.weight.register_hook(get_zero_grad_hook(self.mask_triu, device))

        # with torch.no_grad():
        #     self.L2.weight.copy_(self.U2.weight.permute(0,2,1))
        #     for i in range(self.batch):
        #         self.L2.weight[i].fill_diagonal_(1)
        #     self.L2.weight.copy_(self.L2.weight)
        # self.L2.weight.register_hook(get_zero_grad_hook(self.mask_tril, device))
        # self.L2.weight.register_hook(TransposeHook(self.U2))



        self.poolinglayer2 = nn.ModuleList()
        self.poolinglayer2.append(self.U2)
        # self.poolinglayer2.append(self.L2)

        # self.bn = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(256)
        # self.bn1 = nn.BatchNorm1d(41)
        # self.bn2 = nn.BatchNorm1d(41)

        # 共识基序识别滤波
        conv_weights = torch.rand((256, 4, 5))
        # 将权重限制在[0, 1]范围内
        conv_weights = torch.clamp(conv_weights, 0, 1)
        self.conv1d_layer = nn.Conv1d(in_channels=4, out_channels=256, padding=2, kernel_size=5)
        self.conv1d_layer.weight.data = conv_weights

        return

    def cheb_polynomial_multi(self, laplacian):  # 三阶拉普拉斯算子



        # print('laplacian.shape '+str(laplacian.shape)) #torch.Size([145, 41, 41])
        bat, N, N = laplacian.size()  # [N, N] 512

        x = torch.eye(N).to('cuda:0')  # 创建对角矩阵n*n
        out = x.expand((bat, N, N))  # 扩展维度到b维

        laplacian = laplacian.unsqueeze(1)
        first_laplacian = torch.zeros([bat, 1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = ((2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian).squeeze()
        # multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian], dim=1)
        third_laplacian=third_laplacian+second_laplacian.squeeze()+1
        # multi_order_laplacian=torch.sum(multi_order_laplacian,dim=1)
        # return third_laplacian  # 32x12x4x100x100

        return third_laplacian  # 32x12x4x100x100
        # return multi_order_laplacian  # 32x12x4x100x100

    def L_generation(self, G):  # 拉普拉斯矩阵生成
        # print(torch.sum(G[0][2]))
        degree = torch.sum(G, dim=-1)
        degree_l = tensor_diag(degree) # D
        # print(degree_l.shape) #706x41
        L = degree_l - G
        # 是否归一化（暂时不考虑）
        # print('L[0] '+str(sum(L[0])))
        L = self.cheb_polynomial_multi(L)
        # print(L)
        return G, L

    def space_alignment(self, L1, L2, R1, R2, x_pre1, x_pre2, invertible):
        if not invertible:
            # 这部分的可逆不太好做
            # print(L1.shape) #torch.Size([706, 41, 41])
            # print(R1.shape) #torch.Size([706, 41, 4])
            # print(L1)
            # print(L2)
            R1 = R1.float()
            # print('torch.sum(R1)_before ' + str(torch.sum(R1)))
            x1 = torch.matmul(L1, R1)  # 706x41x4
            # print('torch.sum(R1) ' + str(torch.sum(x1)))

            # 引入64个5x4的一维卷积核，将onehot转变为经过共识基序过滤的表征。
            x1 = self.conv1d_layer(x1.permute(0, 2, 1)).permute(0, 2, 1)  # 增加一维卷积 (是否使用一维卷积？因为它并不好可逆)

            # print('torch.sum(R1)_after ' + str(torch.sum(x1)))

            # print('torch.sum(R2)_before ' + str(torch.sum(R2)))

            x2 = torch.matmul(L2, R2)
            # print('torch.sum(R2)_after ' + str(torch.sum(x2)))

            # 计算孪生损失,需要返回损失
            Loss3 = wasserstein_distance_sum_average(x1, x2)  # 对比拉近两种表示在特征空间之间的距离

            return x1, x2, Loss3
        # 执行可逆过程 （这部分需要修改，反卷积+反池化）
        # print(x_pre1.shape)
        # print(R1.shape)

        # 执行可逆过程 （这部分需要修改，反卷积+反池化）
        # print(x_pre1.shape)
        # print(R1.shape)

        # 利用包含共识知识的一维卷积参数添加扰动（直接使用反卷积或上采样，转置卷积）
        # 输入为经过反可逆交互得到的结果，即利用已学习到的卷积核（256x41x4）将一组形如（706x41x1x256）的张量通过反卷积（256x5x4）等方式转换为形如（706x41x4x1）的张量
        # 对卷积核进行转置操作（4x256x5）

        # 是否需要增加对x1的标准化
        motif_weight = self.conv1d_layer.weight.data.permute(1, 0, 2)

        self.conv1d_layer_inverse = nn.Conv1d(in_channels=256, out_channels=4, padding=2, kernel_size=5)
        self.conv1d_layer_inverse.to(device="cuda:0")
        self.conv1d_layer_inverse.weight.data = motif_weight

        R1 = R1.float()
        # print(self.conv1d_layer_inverse.weight.data.device)
        # print(R1.shape) #100x41x256

        R1 = max_min(R1).to('cuda:0')
        R2 = max_min(R2).to('cuda:0')

        x1 = self.conv1d_layer_inverse(R1.permute(0, 2, 1)).permute(0, 2,
                                                                    1)  # 通过对一维卷积核的转置将256的张量转换为4的张量。期望得到形如（706x41x4的张量）
        # print(R1[1])

        # print(x1[1])
        # # 对x1按行进行归一化
        # x1=row_normalize_softmax(x1)
        # print(x1[1])

        x2 = R2

        # print(x1[0])
        # print(x2[0])

        # print(x1[0])
        # print(x2[0])

        # 添加图结构学习部分
        # x1为经过卷积还原得到的张量（706x41x4），x2为经过卷积还原得到的张量（706x41x256）
        # 这一部分的目的是，通过所有节点的表征，得到相应的图（706x41x41）
        # 考虑：（1）图结构学习。（2）如何引入已知参数（不需要额外引入参数）
        # 采用高斯核函数的方式计算其图结构,基本方式为，（1）计算特征之间的马氏距离。（2）计算其高斯相似性。（3）标准化，阈值筛选，获得核心子结构。这部分的问题在于是计算出两个图还是两者共同决定一个图（都试试）

        sigma = 1.0
        mahalanobis_dist_x1 = mahalanobis_distance(x1)
        similarity_x1 = gaussian_kernel_similarity(mahalanobis_dist_x1, sigma)  # 标签图节点表示得到的重构图

        mahalanobis_dist_x2 = mahalanobis_distance(x2)
        similarity_x2 = gaussian_kernel_similarity(mahalanobis_dist_x2, sigma)  # 结构图节点表示得到的重构图

        # print(similarity_x1[0])
        # print(torch.mean(similarity_x1[0][0]))
        # print('similarity_x2.shape'+str(similarity_x1.shape))
        #
        # # 对x1,先进行阈值筛选，再进行归一化，再进行阈值处理
        # print(similarity_x1[0][0])
        # similarity_x1=threshold_filter_motif(similarity_x1, 2) # （1）在一级结构图中，只观察对角线附近的值；或（2）保留每行中的topk个节点。
        # print(similarity_x1[0][0])

        # # 先进行阈值筛选，再进行归一化，再进行阈值处理
        # print(similarity_x1[0][3])
        # similarity_x1=keep_top_k_values(similarity_x2, 5) # （1）在一级结构图中，只观察对角线附近的值；或（2）保留每行中的topk个节点。
        # print(similarity_x1[0][3])

        # 对x2，直接进行阈值筛选，保留除对角线外的五个最大值

        print(similarity_x1[0])
        similarity_x1 = threshold_filter(similarity_x1)
        similarity_x1 = keep_top_k_values(similarity_x1, 4)
        print(similarity_x1[0])

        print(similarity_x2[0])
        similarity_x2 = threshold_filter(similarity_x2)
        similarity_x2 = keep_top_k_values(similarity_x2, 4)
        print(similarity_x2[0])

        # 返回两个图

        # L1 = solve_for_A(x_pre1, R1)
        # L2 = solve_for_A(x_pre2, R2)  # 计算出两个拉普拉斯算子，该部分的计算复杂度较高。但实际上x为nx1,相对来说也还行。
        #
        # L1 = -torch.sqrt(L1 * 0.5)
        # L2 = -torch.sqrt(L2 * 0.5)  # D-W
        #
        # # 直接将L1, L2的对角线取代为1
        # identity_matrix = torch.eye(L1.size(-1))
        #
        # # 使用逐元素的索引操作和单位矩阵的广播，将单位矩阵的值赋给每个矩阵的对角线元素
        # L1 = L1 * (1 - identity_matrix) + identity_matrix
        # L2 = L2 * (1 - identity_matrix) + identity_matrix

        # LU分解的逆，LU分解，降低torch.linalg.solve的计算复杂度。
        G1 = similarity_x1
        G2 = similarity_x2

        # 存储两类图
        # np.save('./Pre-Encoding/data/Inverse_validation/graph/valid_G1.npy',G1)
        # np.save('./Pre-Encoding/data/Inverse_validation/graph/valid_G2.npy',G2)

        return G1.to(torch.device('cuda:0')), G2.to(torch.device('cuda:0'))

    def forward(self, G1, G2, R1, R2, x_pre1, x_pre2, invertible):  # 输入参数分别为图邻接矩阵1，邻接矩阵2，节点属性表示1（onehot），节点属性2 (ELom4)
        if not invertible:
            # print(R2.shape)
            # print(R1.shape)
            # R1=self.bn1(R1.permute(0,2,1)).permute(0,2,1) #R1不做标准化
            R2=self.bn2(R2.permute(0,2,1)).permute(0,2,1)

            # 加一部分关于图的流
            G1,G2=self.graph_interaction(G1,G2,invertible)

            #手动更新weight并使其保持对称
            for i, layer in enumerate(self.poolinglayer1):
                G1 = layer(G1) #只使用一层。
            for i, layer in enumerate(self.poolinglayer2):
                G2 = layer(G2)

            # print('G1.shape '+str(G1.shape)) #([705, 41, 41])
            # print('G2[1] '+str(G2[1]))

            # 在此处计算图池化的正向损失,损失需要返回
            Loss1=graph_neg_log_likelihood(G1,self.poolinglayer1)
            Loss2=graph_neg_log_likelihood(G2,self.poolinglayer2)

            G1, L1 = self.L_generation(G1)
            G2, L2 = self.L_generation(G2)

            # print('str(L1[0] '+str(L1[0]))
            # print('str(L2[0] '+str(L2[0]))

            # print('L1.shapa '+str(L1.shape)) #torch.Size([705, 41, 41])

            x1, x2, Loss3 = self.space_alignment(L1, L2, R1, R2, R1, R2, invertible)
            return G1, G2, x1, x2, Loss1, Loss2, Loss3

        #逆过程
        print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
        G1, G2 = self.space_alignment(G1, G2, R1, R2, x_pre1, x_pre2, invertible)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # 倒序，可能会报错
        reversed_layers1 = list(reversed(self.poolinglayer1))
        reversed_layers2 = list(reversed(self.poolinglayer2))
        for i, layer in enumerate(reversed_layers1):
            # G1 = torch.linalg.solve(layer.weight, G1)
            G1 = safe_elementwise_divide(G1,layer.weight)
        for i, layer in enumerate(reversed_layers2):
            # G2 = torch.linalg.solve(layer.weight, G2)
            G2 = safe_elementwise_divide(G2,layer.weight)

        # 加一部分关于图的可逆
        G1,G2 = self.graph_interaction(G1, G2, invertible)

        # 保存图结构 (加上图的可逆部分)
        print("执行存储步骤")
        np.save('./Pre-Encoding/data/Inverse_validation/graph/rat_liver/valid_G1_fold'+str(self.fold)+'.npy',G1)
        np.save('./Pre-Encoding/data/Inverse_validation/graph/rat_liver/valid_G2_fold'+str(self.fold)+'.npy',G2)

        return G1, G2

class ScalingLayer(nn.Module): #感觉不用这一层，one-hot的表征会向0靠近。
    def __init__(self, data_dim):
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.randn(1, data_dim, requires_grad=True))

    def forward(self, x, logdet, invert=False):
        log_det_jacobian = torch.sum(self.log_scale_vector)  # 此处的行列式为对角线元素之和（对数）

        if invert:
            return torch.exp(- self.log_scale_vector) * x, logdet - log_det_jacobian

        return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian


class InvertibleBigraphNeuralNetwork(nn.Module):  # 当前为一层的卷积过程,默认均为三维张量
    def __init__(self, num, size, batch, batch1, fold, device="cuda:0"):
        super(InvertibleBigraphNeuralNetwork, self).__init__()

        self.size = size
        self.batch = batch
        self.batch1 = batch1
        self.num = num
        self.fold = fold
        self.laplacian_eneration_layer1 = LaplacianGeneration(self.num, self.batch, self.size)
        # self.laplacian_eneration_layer2 = LaplacianGeneration(self.num, self.batch1, self.size)
        self.siamese_coupling_layer = Siamese_Coupling(self.num)
        self.to(device)
    def forward(self, G1, G2, x1, x2, x_pre1, x_pre2, invertible):
        if not invertible:
            # print('torch.sum(x1)_before '+str(torch.sum(x1)))
            # print('torch.sum(x2)_before '+str(torch.sum(x2)))
            G1, G2, x1, x2, Loss1, Loss2, Loss3 = self.laplacian_eneration_layer1(G1, G2, x1, x2, x_pre1, x_pre2,
                                                            invertible)  # 正向的x_pre1, x_pre2为x1, x2,
            # else:
            #     G1, G2, x1, x2, Loss1, Loss2, Loss3 = self.laplacian_eneration_layer2(G1, G2, x1, x2, x_pre1, x_pre2,
            #                                                                          invertible)  # 正向的x_pre1, x_pre2为x1, x2,
            # print('torch.sum(x1) '+str(torch.sum(x1)))
            # print('torch.sum(x2) '+str(torch.sum(x2)))
            x1, x2 = self.siamese_coupling_layer(x1, x2, invertible) #去掉这部分呢（没差）
            # print(x1[0])
            # print(x2[0])
            #损失需要返回
            # print('torch.sum(x1)_after '+str(torch.sum(x1)))
            # print('torch.sum(x2)_after '+str(torch.sum(x2)))
            # 需要对这部分loss进行一定的限制
            Loss4=representation_neg_log_likelihood(x1,self.siamese_coupling_layer)
            Loss5=representation_neg_log_likelihood(x2,self.siamese_coupling_layer)
            return x1, x2, Loss1, Loss2, Loss3, Loss4, Loss5
        """
            (1) 对齐层可逆。（2）LU层可逆
        """

        x1, x2 = self.siamese_coupling_layer(x1, x2, invertible)  # 逆向,值正常


        G1, G2 = self.laplacian_eneration_layer1(G1, G2, x1, x2, x_pre1, x_pre2,self.fold, invertible)

        return G1, G2
