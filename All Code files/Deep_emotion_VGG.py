import torch
import torch.nn as nn
import torch.nn.functional as F

class Deep_Emotion_VGG(nn.Module):
    def __init__(self):
        super(Deep_Emotion_VGG, self).__init__()

        # VGG16-like architecture
        self.conv1_1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.4)
        self.conv1_2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        # self.conv5_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.conv5_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.pool5 = nn.MaxPool2d(2, 2)

        self.norm = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(1152 , 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 8)  # Assuming 7 classes

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, input):
        out = self.stn(input)

        out = F.relu(self.conv1_1(out))
        out = self.dropout1(out)

        out = F.relu(self.conv1_2(out))
        out = self.dropout1(out)

        out = F.relu(self.pool1(out))

        out = F.relu(self.conv2_1(out))
        out = self.dropout1(out)

        out = F.relu(self.conv2_2(out))
        out = self.dropout1(out)

        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3_1(out))
        out = self.dropout1(out)

        out = F.relu(self.conv3_2(out))
        out = self.dropout1(out)

        out = F.relu(self.conv3_3(out
                                 ))
        out = self.dropout1(out)

        out = F.relu(self.pool3(out))

        out = F.relu(self.conv4_1(out))
        out = self.dropout1(out)

        out = F.relu(self.conv4_2(out))
        out = self.dropout1(out)

        out = F.relu(self.conv4_3(out))
        out = self.dropout1(out)

        out = F.relu(self.pool4(out))

        # out = F.relu(self.conv5_1(out))
        # out = self.dropout1(out)

        # out = F.relu(self.conv5_2(out))
        
        # out = self.dropout1(out)

        # out = F.relu(self.conv5_3(out))
        # out = self.dropout1(out)

        # out = F.relu(self.pool5(out))

        out = self.norm(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = self.dropout1(out)

        out = F.relu(self.fc2(out))
        out = self.dropout1(out)

        out = self.fc3(out)

        return out
