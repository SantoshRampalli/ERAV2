# Import required libraries

import torch
import torch.nn.functional as F
import torch.nn as nn

class Network(nn.Module):

    # Defines NN structure
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3)

        self.fc1 = nn.Linear(in_features=4096,out_features=50)
        self.fc2 = nn.Linear(in_features=50,out_features=10)

    def forward(self,data):
        # Input Layer
        x = data

        # Conv1 Layer
        x = self.conv1(x)
        x = F.relu(input=x)

        # Conv2 Layer
        x = self.conv2(x)
        x = F.max_pool2d(input=x,kernel_size=2)
        x = F.relu(x)

        # Conv3 Layer
        x = self.conv3(x)
        x = F.relu(x)

        # Conv4 Layer
        x = self.conv4(x)
        x = F.max_pool2d(input=x,kernel_size=2)
        x = F.relu(x)

        # 2D to 1D
        x = x.view(-1,4096)

        # FC1 Layer
        x = self.fc1(x)
        x = F.relu(x)

        # FC2 Layer
        x = self.fc2(x)
        x = F.relu(x)

        # Output Layer
        output = F.softmax(input=x,dim=1)
        return output
