import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys

from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import mahalanobis
from scipy.spatial import distance
from sklearn.metrics import roc_curve, roc_auc_score
def visualize_matrix(matrix):
    """
    可视化二维矩阵
    :param matrix: 二维矩阵，作为np数组
    """
    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, matrix)

    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_zlabel('Height')
    plt.show()


def plot_heatmap_surface(matrix):
    # 创建x和y坐标轴
    y = np.arange(0, matrix.shape[1])
    x = np.arange(0, matrix.shape[0])

    x, y = np.meshgrid(x, y)

    plt.subplot(121)

    plt.imshow(matrix, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar()

    plt.title("Spatio-Temporal Dependency")
    plt.xlabel("Temporal Patches")
    plt.ylabel("Temporal Patches")

    # # 绘制曲面图
    # ax = plt.subplot(122, projection='3d')
    # ax.plot_surface(x, y, matrix, cmap='hot')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Value')

    # plt.figure(figsize=(8, 6))
    plt.show()

    # 显示图形



# def mahalanobis_similarity(vectors):
#     num_vectors = vectors.shape[0]
#     covariance = np.cov(vectors.T)
#     inv_covariance = np.linalg.inv(covariance)
#
#     # 计算每对向量之间的马氏距离
#     diff = vectors[:, np.newaxis, :] - vectors
#     distances = np.sqrt(np.einsum('ijk,kl,ijl->ij', diff, inv_covariance, diff))
#
#     # 将马氏距离转换为相似性（相似性越高，距离越小）
#     similarities = 1.0 / (1.0 + distances)
#
#     # 将相似性矩阵的对角线元素设置为0（每个向量与自身的相似性为0）
#     np.fill_diagonal(similarities, 0)
#
#     # 对每个向量的相似性进行求和，并将结果缩放到0-1范围
#     similarities_sum = np.sum(similarities, axis=1)
#     similarities_normalized = similarities_sum / (num_vectors - 1)
#
#     return similarities_normalized



def pearson_similarity(matrix):
    mean = np.mean(matrix, axis=1, keepdims=True)  # 计算每个向量的均值
    centered_matrix = matrix - mean  # 将矩阵中每个向量减去其均值，得到中心化矩阵
    norm = np.linalg.norm(centered_matrix, axis=1)  # 计算每个向量的范数

    # 计算皮尔逊相关系数
    similarity = np.dot(centered_matrix, centered_matrix.T) / (norm.reshape(-1, 1) * norm)

    return np.diag(similarity)  # 返回对角线上的元素，即每个向量的相似性

def plot_multivariate_time_series(data):
    # # 创建子图
    # fig, ax = plt.subplots()
    #
    # # 绘制线性图
    # for i in range(data.shape[1]):
    #     ax.plot(data[:, i], label=f'Series {i+1}')
    #
    # # 添加图例
    # ax.legend()
    #
    # # 添加标题和标签
    # ax.set_title('Multi-Series Linear Plot')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Value')
    #
    # # 显示图形
    # plt.show()

    num_dimensions = data.shape[1]

    # 创建子图
    fig, axs = plt.subplots(num_dimensions, 1, sharex=True)

    # 绘制线性图
    for i in range(num_dimensions):
        axs[i].plot(data[:, i])
        axs[i].set_ylabel(f'Series {i+1}')

    # 添加标题和共享的x轴标签
    axs[-1].set_xlabel('Time')
    fig.suptitle('Multivariate Time Series')

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.3)

    # 显示图形
    plt.show()


# def mahalanobis_distributions(vector_set):
#     # 计算向量组的均值和协方差矩阵
#     mean = np.mean(vector_set, axis=0)
#     cov_matrix = np.cov(vector_set.T)
#
#     # 计算协方差矩阵的逆矩阵
#     cov_inv = np.linalg.inv(cov_matrix)
#
#     # 将向量组重塑为二维数组
#     vector_set_2d = np.reshape(vector_set, (vector_set.shape[0], -1))
#
#     # 计算向量组每个向量到向量组的马氏距离
#     distances_to_set = distance.cdist(vector_set_2d, [mean], metric='mahalanobis', VI=cov_inv)
#
#     # 计算向量组每个向量在向量组中的分布情况
#     distributions = np.exp(-0.5 * distances_to_set**2) / (np.sqrt((2 * np.pi)**vector_set.shape[1] * np.linalg.det(cov_matrix)))
#
#     return distributions.flatten()


def mahalanobis_distance(X, cov=None, mu=None):
    if mu is None:
        mu = np.mean(X, axis=0)
    if cov is None:
        cov = np.cov(X, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError as err:
        print("Error, probably singular matrix!")
        inv_cov = np.eye(cov.shape[0])
    X_diff_mu = X - mu
    M = np.apply_along_axis(lambda x:
                               np.matmul(np.matmul(x, inv_cov), x.T), 1, X_diff_mu)
    return M


def plot_heatmap(matrix):
    """
    绘制热力图

    参数:
    matrix -- 输入的二维矩阵

    返回:
    无
    """
    # 获取矩阵的行数和列数
    M, N = matrix.shape

    # 创建绘图对象
    fig, ax = plt.subplots()

    # 绘制热力图
    im = ax.imshow(matrix)
    # for i in range(matrix.shape[0]):
    #     for j in range(matrix.shape[1]):
    #         plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white')

    # 设置刻度标签
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(M))

    # 设置图形标题和坐标轴标签
    if M==N :
        ax.set_title("Spatio-Temporal Dependency",fontsize=18)
        ax.set_xlabel("Temporal Patches",fontsize=18)
        ax.set_ylabel("Temporal Patches",fontsize=18)
    else:
        ax.set_title("Temporal Patches",fontsize=18)
        ax.set_xlabel("Tp",fontsize=18)
        ax.set_ylabel("S",fontsize=18)
    plt.colorbar(im)
    # 显示图形
    # plt.show()




def plot_multivariate_timeseries(data):
    """
    绘制多元时间序列的函数

    参数:
    - data: 形状为 [M, N, K] 的矩阵，表示多元时间序列数据

    注意:
    - M 表示序列分成的行数
    - N 表示序列分成的列数
    - K 表示每个序列的长度
    - 假设 M * N <= K，否则会有部分数据被忽略
    - 每个画布不用颜色，使用线条表示
    """

    M, N, K = data.shape
    fig, axs = plt.subplots(M, N, sharex=True, sharey=True, figsize=(11, 10))  # 创建画布，大小为 M * N
    fig.subplots_adjust(hspace=0.1)  # 调整子图之间的垂直间距
    plt.subplots_adjust(wspace=0.1)

    for i in range(M):
        for j in range(N):
            sequence = data[i, j]  # 获取当前序列的数据
            axs[i, j].plot(range(K), sequence, color='blue')  # 绘制当前序列的线条
            axs[i, j].set_title(f'Patches {i}, {j}')  # 设置子图标题

    plt.tight_layout()  # 调整子图的布局
    plt.show()  # 显示图像

def oringe_time_series(data):
    M, N = data.shape

    # 计算子画布的布局
    rows = N
    cols = 1

    # 调整画布尺寸
    fig, axs = plt.subplots(rows, cols, figsize=(8, 3 * rows), sharex=True)

    # 绘制每个维度的多元时间序列
    for i in range(N):
        ax = axs[i][0]

        # 绘制当前维度的多元时间序列
        ax.plot(range(M), data[:, i], color='black', linewidth=1)

        # 设置子画布的标题和坐标轴标签，并调整字体大小
        ax.set_title('Dimension {}'.format(i + 1), fontsize=8)
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)

        # 设置刻度标签的字体大小
        ax.tick_params(axis='both', labelsize=6)

        # 移除子画布的边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # 移除刻度标签之间的间距
        ax.margins(x=0)

        # 移除子画布的背景色
        ax.set_facecolor('none')

    # 调整子画布的间距
    plt.subplots_adjust(hspace=0.05)

    # 隐藏画布的边框
    fig.set_frameon(False)

    # 显示图形
    plt.show()



def With_labels(data, labels):

    M, N = data.shape

    # 计算子画布的布局
    rows = N
    cols = 1

    # 调整画布尺寸
    fig, axs = plt.subplots(rows, cols, figsize=(8, 3 * rows), sharex=True)

    # 绘制每个维度的多元时间序列和标签
    for i in range(N):
        ax = axs[i]

        # 绘制当前维度的多元时间序列
        ax.plot(range(M), data[:, i], color='black', linewidth=3)

        # 绘制标签为1的时间戳的红色半透明矩形
        for j in range(M):
            if labels[j] == 1:
                rect = plt.Rectangle((j - 0.5, ax.get_ylim()[0]), 1, ax.get_ylim()[1] - ax.get_ylim()[0], color='red',
                                     alpha=0.5)
                ax.add_patch(rect)

        # 设置子画布的标题和坐标轴标签，并调整字体大小
        # ax.set_title('Dim {}'.format(i + 1), fontsize=8)
        # ax.set_xlabel('Time', fontsize=8)
        # ax.set_ylabel('Value', fontsize=15)

        # 设置刻度标签的字体大小
        ax.tick_params(axis='both', labelsize=6)

        # 移除子画布的边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # 移除刻度标签之间的间距
        ax.margins(x=0)

        # 移除子画布的背景色
        ax.set_facecolor('none')
    fig.suptitle('Multivariate Timeseries', fontsize=38)
    # fig.text(0.06, 0.5, 'Shared Y-axis Label', va='center', rotation='vertical')
    plt.xlabel('Time', fontsize=38)

    # 调整子画布的间距
    plt.subplots_adjust(hspace=0.05)

    # 隐藏画布的边框
    fig.set_frameon(False)

    # 显示图形



def plot_roc_curve(y_true, y_scores):
    """
    绘制ROC曲线

    参数:
    - y_true: 真实标签数组
    - y_scores: 预测分数数组
    """

    # 计算TPR和FPR
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # 计算AUC值
    auc = roc_auc_score(y_true, y_scores)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # 显示图形
    plt.show()



# random_matrix = np.random.random([100,9])
# random_label = np.random.randint(2, size=[100])
# With_labels(random_matrix,random_label)


