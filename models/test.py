import torch

def mahalanobis_distance(X, regularization_param=1e-5):
    # 计算均值向量
    mean_vector = torch.mean(X, dim=1, keepdim=True)

    # 计算去中心化矩阵
    X_c = X - mean_vector

    # 计算协方差矩阵
    cov_matrix = torch.matmul(X_c.transpose(1, 2), X_c) / (X.size(1) - 1)

    # 添加正则化项
    cov_matrix += regularization_param * torch.eye(cov_matrix.size(-1), device=cov_matrix.device)

    # 计算协方差矩阵的逆矩阵
    inv_cov_matrix = torch.inverse(cov_matrix)

    print(X_c.shape)
    print(inv_cov_matrix.shape)
    print(X_c.transpose(1, 2).shape)

    # 计算马氏距离
    mahalanobis_dist = torch.bmm(torch.bmm(X_c, inv_cov_matrix), X_c.transpose(1, 2))

    return mahalanobis_dist.squeeze()

def gaussian_kernel_similarity(mahalanobis_dist, sigma=1.0):
    # 计算高斯核函数
    similarity = torch.exp(-0.5 * (mahalanobis_dist / sigma)**2)

    return similarity

# 示例
# 创建输入张量 [b, n, m]
batch_size = 706
num_samples = 41
num_features = 256
input_tensor = torch.randn(batch_size, num_samples, num_features)

# 计算马氏距离（带正则化）
mahalanobis_dist = mahalanobis_distance(input_tensor)
# 基于高斯核函数计算相似距离
sigma = 1.0
similarity = gaussian_kernel_similarity(mahalanobis_dist, sigma)

print("Mahalanobis Distance Shape:", mahalanobis_dist.shape)
print("Mahalanobis Distance:")
print(mahalanobis_dist)

print("\nGaussian Kernel Similarity:")
print(similarity) # torch.Size([706, 41, 41])

print(similarity.shape)
