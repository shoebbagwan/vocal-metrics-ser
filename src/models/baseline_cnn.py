import torch
import torch.nn as nn

class BaselineCNN1D(nn.Module):
    """
    1D Convolutional Neural Network for Speech Emotion Recognition.
    Designed to take MFCC features as input: shape (Batch, 40, TimeSteps)
    """
    def __init__(self, num_classes: int = 8, input_channels: int = 40):
        super(BaselineCNN1D, self).__init__()
        
        # 1st Convolutional Block
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(0.3)
        
        # 2nd Convolutional Block
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(0.3)
        
        # 3rd Convolutional Block
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.drop3 = nn.Dropout(0.3)
        
        # Global Average Pooling to handle variable length (though we padded our inputs)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected Classifier
        self.fc1 = nn.Linear(256, 128)
        self.fc_relu = nn.ReLU()
        self.fc_drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x is expected to be of shape (Batch, Num_MFCC, TimeSteps)
        x = self.drop1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        
        x = self.fc_drop(self.fc_relu(self.fc1(x)))
        out = self.fc2(x)
        
        return out
