import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from layers import DenseCaps, PrimaryCaps


class CapsuleNet(nn.Module):
    """
    Input: (batch, channels, width, height)
    Output:((batch, classes), (batch, channels, width, height))

    input_size: [channels, width, height]
    classes: number of classes
    iterations：dynamic routing iterations
    """
    def __init__(self, input_size, classes, iterations):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.iterations = iterations

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCaps(256, 256, 8, kernel_size=9, stride=2, padding=0)

        # Layer 3: Capsule layer. iterative dynamic routing.
        self.digitcaps = DenseCaps(in_num_caps=32*6*6, in_dim_caps=8,
                                      out_num_caps=classes, out_dim_caps=16, iterations=iterations)

        # reconstruction net
        self.reconstructor = nn.Sequential(
            nn.Linear(16*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()


    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)  # [batch, out_num_caps, out_dim_caps]
        length = x.norm(dim=-1)  # vector lenght代表存在概率 [batch, out_num_caps, 1]

        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            # 将index处，更改为1
            y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())

        # y[:, :, None]: mask
        reconstruction = self.reconstructor((x * y[:, :, None]).view(x.size(0), -1))
        # 存在概率预测，重构图像像素
        return length, reconstruction.view(-1, *self.input_size)