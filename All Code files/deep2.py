import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(3,10,3)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(10,10,3)
        self.dropout2 = nn.Dropout(0.25)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.dropout3 = nn.Dropout(0.25)
        self.conv4 = nn.Conv2d(10,10,3)
        self.dropout4 = nn.Dropout(0.25)
        self.pool4 = nn.MaxPool2d(2,2)
        
        self.conv5 = nn.Conv2d(50,50,2)
        self.conv6 = nn.Conv2d(50,10,2)
        self.pool6 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810,100)
        self.fc2 = nn.Linear(100,8)
        self.softmax = nn.Softmax(dim = 1)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(15, 10, kernel_size=5),
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

    def forward(self,input):
        #out = self.stn(input)
        out2 = self.stn(input)
        #print(out2.shape)
        out = F.relu(self.conv1(out2))
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.dropout3(out)

        out = self.norm(self.conv4(out))
        out = self.dropout4(out)

        out = F.relu(self.pool4(out))
        #print(out.shape)
        ''' out = F.relu(self.conv5(out))
        out = self.conv6(out)
        out = F.relu(self.pool6(out))
        #print(out.shape)'''
        out = F.dropout(out)
        #print(out.shape)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.softmax(out)
       # out = np.argmax(out)

        return out
