import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionCNN(nn.Module):
    def __init__(self, num_classes, num_attention_heads=4):
        super(SelfAttentionCNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3,8,3)
        self.dropout1 = nn.Dropout(0.15)
        self.conv2 = nn.Conv2d(8,16,3)
        self.dropout2 = nn.Dropout(0.15)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(16,24,3)
        self.dropout3 = nn.Dropout(0.15)
        self.conv4 = nn.Conv2d(24,32,3)
        self.dropout4 = nn.Dropout(0.15)
        #self.pool4 = nn.MaxPool2d(2,2)

        # Define the attention mechanism
        self.self_attention = nn.MultiheadAttention(embed_dim=32, num_heads=num_attention_heads)
        self.flattened_tensor = nn.Flatten()


        # Define fully connected layers for classification
        self.fc1 = nn.Linear(324 , 32)
        self.fc2 = nn.Linear(32, 8)
        self.softmax = nn.Softmax(dim=1)  # Softmax layer for class probabilities

    def forward(self, x):
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x= self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x= self.dropout3(x)
        x = torch.relu(self.conv4(x))
        x= self.dropout4(x)


        x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)  # Reshape to (sequence_length, batch_size, features)

        # Apply self-attention mechanism
        x, _ = self.self_attention(x, x, x)

        # Reshape back to the original dimensions using reshape
        x = x.permute(1, 2, 0).reshape(x.size(0), -1, x.size(-1))

        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)

        # Fully connected layers for classification
        x = x.view(-1, 324)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        # Apply softmax for class probabilities
        x = self.softmax(x)

        return x

# Instantiate the model
