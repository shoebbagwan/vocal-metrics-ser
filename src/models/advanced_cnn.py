import torch
import torch.nn as nn

class AdvancedCNNLSTM(nn.Module):
    """
    Advanced Model: 2D CNN followed by LSTM for temporal dynamics.
    Designed for Mel Spectrogram inputs of shape: (Batch, 1, Mel_Bins, TimeSteps)
    """
    def __init__(self, num_classes: int = 8, input_channels: int = 1):
        super(AdvancedCNNLSTM, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.3)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(0.3)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout(0.3)
        
        # LSTM layer
        # After 3 maxpools of (2,2), the Mel dimension (usually 128) becomes 128 / 8 = 16
        # Channel dim is 128. So we flatten these into: 128 * 16 = 2048 features per timestep
        self.lstm_input_size = 128 * 16
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, 
                            hidden_size=256, 
                            num_layers=2, 
                            batch_first=True, 
                            dropout=0.3,
                            bidirectional=True)
                            
        # Classifier
        self.fc1 = nn.Linear(256 * 2, 128) # *2 for bidirectional
        self.fc_relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x is (Batch, Channels=1, Mel_Bins, TimeSteps)
        x = self.dropout1(self.maxpool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.maxpool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.maxpool3(self.relu3(self.bn3(self.conv3(x)))))
        
        # Now x is (Batch, 128, Mel_Bins/8, TimeSteps/8)
        # We need to rearrange this to (Batch, TimeSteps/8, 128 * Mel_Bins/8) for LSTM
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # (Batch, TimeSteps, Channels, Mels)
        x = x.view(B, W, C * H) # Flatten channels and Mel dimension
        
        # LSTM pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # We take the output of the last timestep
        last_out = lstm_out[:, -1, :] 
        
        # Classifier
        x = self.fc_dropout(self.fc_relu(self.fc1(last_out)))
        out = self.fc2(x)
        
        return out
