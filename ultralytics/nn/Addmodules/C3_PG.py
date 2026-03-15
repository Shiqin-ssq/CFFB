import torch
from scipy.special import cosm1
from torch import nn
from ultralytics.nn.modules import Conv


def drop_path(x, drop_prob=0.0, training=False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956  ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # PyTorch uses x.ndim to get the number of dimensions
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # Ensure the random tensor is on the same device as x
    random_tensor = torch.floor(random_tensor)  # binarize
    output = x * random_tensor / keep_prob  # In PyTorch, we multiply before dividing to maintain numerical stability
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PartialConv(nn.Module):
    def __init__(self, dim, n_div=4, kernel_size=3, forward='split_cat'):
        """
        PartialConv 模块
        Args:
            dim (int): 输入张量的通道数。
            n_div (int): 分割通道数的分母，用于确定部分卷积的通道数。
            forward (str): 使用的前向传播方法，可选 'slicing' 或 'split_cat'。
        """
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class FasterNetBlock(nn.Module):
    def __init__(self, dim, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0, forward='split_cat'):
        super().__init__()
        self.pconv = PartialConv(dim, forward=forward)
        self.conv1 = nn.Conv2d(dim, dim * expand_ratio, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim * expand_ratio)
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(dim * expand_ratio, dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.pconv(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = residual + self.drop_path(x)
        return x


# class C3_PG(nn.Module):
#     def __init__(self, c1, c2, e=0.5):
#         super().__init__()
#         e = 0.5
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, self.c, 1, 1)
#         self.cv2 = PGConv(self.c, self.c)
#         self.cv3 = Conv(self.c, self.c, 1, 1)
#         self.cv4 = Conv(2 * self.c, c2, 1, 1)
#
#     def forward(self, x):
#         #print(f"c1 :{self.c1}")
#         #print(f"c2 :{self.c2}")
#         #print(f"c:{self.c}")
#         x1 = self.cv1(x)
#         #print(f"x1 shape: {x1.shape}")
#         x2 = self.cv2(self.cv2(x1))
#         #print(f"x2 shape: {x2.shape}")
#         x3 = self.cv3(x1)
#         #print(f"x3 shape: {x3.shape}")
#         y = torch.concat((x2, x3), 1)
#         #print(f"y shape: {y.shape}")
#         return self.cv4(y)