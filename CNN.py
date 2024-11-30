import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, filters=None):  
        if filters is None:
            filters=[32,64,128]
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=filters[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.bn3 = nn.BatchNorm2d(filters[2])
        
        self.dropout = nn.Dropout(0.5)  # 50% dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        
        self.fc1 = nn.Linear(filters[2] * 3 * 3, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout after FC1
        x = self.fc2(x)  # Output layer
        return x
