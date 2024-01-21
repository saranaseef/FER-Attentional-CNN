import torch
import torch.nn as nn
import torch.nn.functional as F

class GridAttention(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size):
        super(GridAttention, self).__init__()
        self.grid_size = grid_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.attention = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        attention_map = F.sigmoid(self.attention(x))
        return x * attention_map

class CNNWithGridAttention(nn.Module):
    def __init__(self, num_classes, grid_size):
        super(CNNWithGridAttention, self).__init__()
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.grid_attention1 = GridAttention(8, 8, grid_size)
        self.conv2 = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.grid_attention2 = GridAttention(12, 16, grid_size)
        self.conv3 = nn.Conv2d(16,24, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc_input_size = 512 * (256 // (2**len(grid_size))) * (256 // (2**len(grid_size)))

        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.grid_attention1(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.grid_attention2(F.relu(self.conv2(x)))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 4608)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
num_classes = 10  # Assuming you have 10 classes for classification
grid_size = (2, 2)  # You can adjust the grid size as needed
model = CNNWithGridAttention(num_classes, grid_size)

# Print the model architecture
print(model)
