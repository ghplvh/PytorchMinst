"""
@file:simpleNet.py
@time:2020/2/5-15:25
"""
from torch import nn, optim


class simpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        """
        :param in_dim: 输入的维度
        :param n_hidden_1: 第一层网络的神经元个数
        :param n_hidden_2: 第二层网络的神经元个数
        :param out_dim: 第三层网络的神经元个数
        """
        super(simpleNet, self).__init__()
        # 1、添加激活函数增加网络的非线性
        # 2、批标准化--加快收敛速度
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        # 最后一层是实际得分
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

#     # 图片维度28*28 隐藏层300、100 节点 输出0-9这10个数，所以是10分类
#     model = simpleNet(28 * 28, 300, 100, 10)
