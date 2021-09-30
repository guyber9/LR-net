from __future__ import print_function
import torch
from torch.nn import functional as F
import torch.nn as nn
import layers as lrnet_nn

################
## MNIST nets ##
################

class FPNet(nn.Module):

    def __init__(self):
        super(FPNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)  # 32 x 24 x 24
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)  # 32 x 12 x 12
        x = F.relu(x)
        x = self.conv2(x)  # 64 x 8 x 8
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)  # 64 x 4 x 4
        x = F.relu(x)
        x = torch.flatten(x, 1)  # 1024
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x
        return output


class LRNet(nn.Module):

    def __init__(self):
        super(LRNet, self).__init__()
        self.conv1 = lrnet_nn.LRnetConv2d(1, 32, 5, 1)
        self.conv2 = lrnet_nn.LRnetConv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)  # 32 x 24 x 24
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)  # 32 x 12 x 12
        x = F.relu(x)
        x = self.conv2(x)  # 64 x 8 x 8
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)  # 64 x 4 x 4
        x = F.relu(x)
        x = torch.flatten(x, 1)  # 1024
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x
        return output

    def train_mode_switch(self):
        self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)


##################
## CIFAR10 nets ##
##################

class FPNet_CIFAR10(nn.Module):

    def __init__(self):
        super(FPNet_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 3
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)  # 128 x 32 x 32
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)  # 128 x 16 x 16
        x = F.relu(x)

        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)  # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2)  # 256 x 8 x 8
        x = F.relu(x)

        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)  # 512 x 8 x 8
        x = self.bn6(x)
        x = F.max_pool2d(x, 2)  # 512 x 4 x 4 (= 8192)
        x = F.relu(x)

        x = torch.flatten(x, 1)  # 8192
        x = self.dropout1(x)
        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # 1024 -> 10
        output = x
        return output


class LRNet_CIFAR10(nn.Module):

    def __init__(self):
        super(LRNet_CIFAR10, self).__init__()
        self.conv1 = lrnet_nn.LRnetConv2d(3, 128, 3, 1, padding=1)
        self.conv2 = lrnet_nn.LRnetConv2d(128, 128, 3, 1, padding=1)
        self.conv3 = lrnet_nn.LRnetConv2d(128, 256, 3, 1, padding=1)
        self.conv4 = lrnet_nn.LRnetConv2d(256, 256, 3, 1, padding=1)
        self.conv5 = lrnet_nn.LRnetConv2d(256, 512, 3, 1, padding=1)
        self.conv6 = lrnet_nn.LRnetConv2d(512, 512, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout2 = nn.Dropout(0.5)

        self.dropout4 = nn.Dropout(0.3)  # 0.2 was 93.13
        self.dropout6 = nn.Dropout(0.3)  # 0.2 was 93.13
        self.dropout1 = nn.Dropout(0.3)

        self.dropout3 = nn.Dropout(0.25)  # 0.2 was 93.13
        self.dropout5 = nn.Dropout(0.25)  # 0.2 was 93.13
        self.dropout7 = nn.Dropout(0.25)  # 0.2 was 93.13

    def forward(self, x):
        x = self.conv1(x)  # input is 3 x 32 x 32, output is 128 x 32 x 32
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout3(x)  # ???
        x = self.conv2(x)  # 128 x 32 x 32
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)  # 128 x 16 x 16
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.conv3(x)  # 256 x 16 x 16
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout5(x)  # ???
        x = self.conv4(x)  # 256 x 16 x 16
        x = self.bn4(x)
        x = F.max_pool2d(x, 2)  # 256 x 8 x 8
        x = F.relu(x)
        x = self.dropout6(x)
        x = self.conv5(x)  # 512 x 8 x 8
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout7(x)  # ???
        x = self.conv6(x)  # 512 x 8 x 8
        x = self.bn6(x)
        x = F.max_pool2d(x, 2)  # 512 x 4 x 4 (= 8192)
        x = F.relu(x)

        x = torch.flatten(x, 1)  # 8192
        x = self.dropout1(x)
        x = self.fc1(x)  # 8192 -> 1024
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # 1024 -> 10
        output = x
        return output

    def train_mode_switch(self):
        self.conv1.train_mode_switch()
        self.conv2.train_mode_switch()
        self.conv3.train_mode_switch()
        self.conv4.train_mode_switch()
        self.conv5.train_mode_switch()
        self.conv6.train_mode_switch()

    def test_mode_switch(self, options, tickets):
        self.conv1.test_mode_switch(options, tickets)
        self.conv2.test_mode_switch(options, tickets)
        self.conv3.test_mode_switch(options, tickets)
        self.conv4.test_mode_switch(options, tickets)
        self.conv5.test_mode_switch(options, tickets)
        self.conv6.test_mode_switch(options, tickets)



















