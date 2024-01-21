import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(SqueezeExciteBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        squeeze = self.squeeze(x)
        squeeze = squeeze.view(x.size(0), -1)
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)
        excitation = excitation.view(x.size(0), -1, 1, 1)
        scaled_input = x * excitation
        return scaled_input

class SECNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SECNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 8, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.1)

        self.se1 = SqueezeExciteBlock(8)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.1)

        self.se2 = SqueezeExciteBlock(8)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.1)

        self.se3 = SqueezeExciteBlock(16)

        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout(0.1)

        self.se4 = SqueezeExciteBlock(32)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.se1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.dropout2(x)

        x = self.se2(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.dropout3(x)

        x = self.se3(x)
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.dropout4(x)

        x = self.se4(x)
        x = self.pool(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x