import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
import numpy as np
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

class LRnetConv2d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            clusters: int = 3,
            transposed: bool = True,
            test_forward: bool = False,
            output_sample: bool = True,
            binary_mode: bool = False,
            eps: int = 1e-07,
    ):
        super(LRnetConv2d, self).__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.clusters = in_channels, out_channels, kernel_size, stride, padding, dilation, groups, clusters
        self.test_forward, self.output_sample, self.binary_mode = test_forward, output_sample, binary_mode
        self.transposed = transposed
        self.eps = eps
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.tensor_dtype = torch.float32

        if self.transposed:
            D_0, D_1, D_2, D_3 = out_channels, in_channels, kernel_size, kernel_size
        else:
            D_0, D_1, D_2, D_3 = in_channels, out_channels, kernel_size, kernel_size

        self.alpha = torch.nn.Parameter(
            torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensor_dtype, device=self.device))
        self.betta = torch.nn.Parameter(
            torch.empty([D_0, D_1, D_2, D_3, 1], dtype=self.tensor_dtype, device=self.device))
        self.test_weight = torch.empty([D_0, D_1, D_2, D_3], dtype=torch.float32, device=self.device)
        self.bias = torch.nn.Parameter(torch.empty([out_channels], dtype=self.tensor_dtype, device=self.device))


        discrete_prob = np.array([-1.0, 0.0, 1.0])
        discrete_prob = np.tile(discrete_prob,
                                [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, 1])
        self.discrete_mat = torch.as_tensor(discrete_prob, dtype=self.tensor_dtype, device=self.device)
        self.discrete_square_mat = self.discrete_mat * self.discrete_mat

        self.cntr = 0
        self.sigmoid = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.alpha, mean=0.0, std=1)
        torch.nn.init.normal_(self.betta, mean=0.0, std=1)
        if self.bias is not None:
            prob_size = torch.cat(((1 - self.alpha - self.betta), self.alpha, self.betta), 4)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(prob_size)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def initialize_weights(self, alpha, betta) -> None:
        print("Initialize Weights")
        with torch.no_grad():
            self.alpha.copy_(torch.from_numpy(alpha))
            self.betta.copy_(torch.from_numpy(betta))

    def train_mode_switch(self) -> None:
        self.test_forward = False

    def test_mode_switch(self, num_of_options=1, tickets=1) -> None:
        with torch.no_grad():
            self.test_forward = True
            sigmoid_func = torch.nn.Sigmoid()
            alpha_prob = sigmoid_func(self.alpha)
            betta_prob = sigmoid_func(self.betta) * (1 - alpha_prob)
            prob_mat = torch.cat(((1 - alpha_prob - betta_prob), alpha_prob, betta_prob), 4)
            m = torch.distributions.Categorical(prob_mat)
            sampled = m.sample()
            values = sampled - 1
            self.test_weight = torch.tensor(values, dtype=self.tensor_dtype, device=self.device)

    def forward(self, input: Tensor) -> Tensor:
        if self.test_forward:
            return F.conv2d(input, self.test_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:

            if torch.isnan(self.alpha).any():
                print("alpha isnan: " + str(torch.isnan(self.alpha).any()))
            if torch.isnan(self.betta).any():
                print("betta isnan: " + str(torch.isnan(self.betta).any()))

            prob_alpha = self.sigmoid(self.alpha)
            prob_betta = self.sigmoid(self.betta) * (1 - prob_alpha)
            prob_mat = torch.cat(((1 - prob_alpha - prob_betta), prob_alpha, prob_betta), 4)
            # E[X] calc
            mean_tmp = prob_mat * self.discrete_mat
            mean = torch.sum(mean_tmp, dim=4)
            m = F.conv2d(input, mean, self.bias, self.stride, self.padding, self.dilation, self.groups)
            # E[x^2]
            mean_square_tmp = prob_mat * self.discrete_square_mat
            mean_square = torch.sum(mean_square_tmp, dim=4)
            # E[x] ^ 2
            mean_pow2 = mean * mean
            sigma_square = mean_square - mean_pow2

            z1 = F.conv2d((input * input), sigma_square, None, self.stride, self.padding, self.dilation, self.groups)
            z1 = z1 + self.eps  ##TODO
            v = torch.sqrt(z1)

            if self.output_sample:
                epsilon = torch.normal(0, 1, size=z1.size(), dtype=self.tensor_dtype, requires_grad=False,
                                       device=self.device)
                return m + epsilon * v
            else:
                return m, v




