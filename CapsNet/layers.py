import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash(inputs, axis=-1):
    """capsule输出的激活函数"""
    norm = torch.norm(inputs, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
    return scale * inputs


class PrimaryCaps(nn.Module):
    """计算第一层capsules的输入，转换成32*6*6个8维的capsule vector
    in_channels：原文中256
    out_channels：卷积后的通道数，原文中256
    dim_caps: PrimaryCaps输出的每个capsule的维度
    kernel_size：原文中9 * 9
    stride：2
    """
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(PrimaryCaps, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)

    def forward(self, input):
        """转换成32*6*6个8维的capsule vector, output size=[batch_size, num_caps, dim_caps]"""
        output = self.conv2d(input)
        output = output.view(input.size(0), -1, self.dim_caps)
        return squash(output)


class DenseCaps(nn.Module):
    """iterative dynamic routing计算capsule目标识别结果vector。
    input size = [None, in_num_caps, in_dim_caps]，
    output size = [None, out_num_caps, out_dim_caps]。

    in_num_caps: 第一层的输入capsule数量，32*6*6
    in_dim_caps：第一层的输入capsule维度，8
    out_num_caps：iterative dynamic routing时及输出的capsule数量，10
    out_dim_caps：iterative dynamic routing时及输出的capsule维度，16
    iterations：dynamic routing轮次
    weight：由32*6*6个8维的capsule vector计算10个16维的capsule vector的transform matrix，在每个[6 * 6]
            单元内的capsule是共享权重的。
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, iterations=3):
        super(DenseCaps, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.iterations = iterations
        self.weight = nn.Parameter(0.01 * torch.randn(1,
                                                      in_num_caps,
                                                      out_num_caps * out_dim_caps,
                                                      in_dim_caps,
                                                      1))

    def forward(self, u):
        """u_hat在不同layer的capsules之间传递，每层capsules只能是才c，b在更新。文中结构只接上了一层
        dynamic routing capsules layer。"""
        # self.weight * u
        #     [1    , in_num_caps, out_num_caps * out_dim_caps, in_dim_caps, 1]
        #     [batch, in_num_caps, out_num_caps * out_dim_caps, in_dim_caps, 1]
        # =>> [batch, in_num_caps, out_num_caps * out_dim_caps, in_dim_caps, 1]
        # 按元素相乘，然后在reduce sum
        u_hat = u[:, :, None, :, None]
        u_hat = self.weight * u_hat.repeat(1, 1, self.out_num_caps * self.out_dim_caps, 1, 1)
        u_hat = torch.sum(u_hat, dim=3)
        # [batch, in_num_caps, out_num_caps, out_dim_caps]
        u_hat = torch.squeeze(u_hat.view(-1,
                                         self.in_num_caps,
                                         self.out_num_caps,
                                         self.out_dim_caps,
                                         1))
        u_hat_for_route = u_hat.detach()

        # coupling coefficient initialize
        # [batch, in_num_caps, out_num_caps]
        b = Variable(torch.zeros(u.size(0), self.in_num_caps, self.out_num_caps)).cuda()
        for i in range(self.iterations):
            c = F.softmax(b, dim=2)  # [batch, in_num_caps, out_num_caps]
            if i < self.iterations - 1:
                # u   [batch, in_num_caps, out_num_caps, out_dim_caps]
                # c   [batch, in_num_caps, out_num_caps, 1]
                # =>> [batch, 1, out_num_caps, out_dim_caps]
                outputs = squash(torch.sum(torch.unsqueeze(c, 3) * u_hat_for_route, dim=1, keepdims=True))
                b = b + torch.sum(outputs * u_hat_for_route, dim=-1)
            else:
                # 此时进入bp计算
                outputs = squash(torch.sum(torch.unsqueeze(c, 3) * u_hat, dim=1, keepdims=True))

        # [batch, out_num_caps, out_dim_caps]
        return torch.squeeze(outputs, dim=1)


def caps_loss(y_true, y_pred, x, x_reconstruct, lamada):
    """Capsule loss = Margin loss + lamada * reconstruction loss.
    y shape [batch, classes], x shape [batch, channels, height, width]"""
    L = y_true * torch.clamp(0.9 - y_pred, min=0) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0) ** 2

    L_margin = L.sum(dim=1).mean()
    L_recon = nn.MSELoss()(x_reconstruct, x)

    return L_margin + lamada * L_recon