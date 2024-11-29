import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.layer_norm1 = nn.LayerNorm([32, 28, 28])  # Layer normalization for conv1 output
        self.layer_norm2 = nn.LayerNorm([64, 14, 14])  # Layer normalization for conv2 output
        self.layer_norm3 = nn.LayerNorm([128, 7, 7])   # Layer normalization for conv3 output
        
        self.dropout = nn.Dropout(0.5)  # 50% dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        
        self.fc1 = nn.Linear(128 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes
        
    def forward(self, x):
        x = self.pool(F.relu(self.layer_norm1(self.conv1(x))))  # Conv1 -> ReLU -> LayerNorm -> Pool
        x = self.pool(F.relu(self.layer_norm2(self.conv2(x))))  # Conv2 -> ReLU -> LayerNorm -> Pool
        x = self.pool(F.relu(self.layer_norm3(self.conv3(x))))  # Conv3 -> ReLU -> LayerNorm -> Pool
        
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout after FC1
        x = self.fc2(x)  # Output layer
        return x
