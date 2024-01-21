import torch
import torch.nn as nn

class RegionBasedAttentionCNN(nn.Module):
    def __init__(self, num_classes, num_attention_regions=4):
        super(RegionBasedAttentionCNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)

        # Define the attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.Softmax(dim=2)  # Apply softmax to select regions
        )

        # Define fully connected layers for classification
        self.fc1 = nn.Linear(64 , 64)
        self.fc2 = nn.Linear(64, 8)
        self.softmax = nn.Softmax(dim=1)  # Softmax layer for class probabilities

    def forward(self, x):
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Apply attention mechanism to extract region-based features
        attention_map = self.attention(x)
        region_features = torch.sum(x * attention_map, dim=(2, 3))

        # Flatten the region-based features
        region_features = region_features.view(region_features.size(0), -1)

        # Fully connected layers for classification
        x = torch.relu(self.fc1(region_features))
        x = self.fc2(x)

        # Apply softmax for class probabilities
        x = self.softmax(x)

        return x

# Instantiate the model
