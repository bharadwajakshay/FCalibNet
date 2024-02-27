import torch.nn as nn
import torch

class transRegression(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
        channels = 128
    
        self.conv1 = nn.Conv1d(channels,64,kernel_size=1,padding='valid',bias=False)
        self.batchNorm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64,32,kernel_size=1,padding='valid',bias=False)
        self.batchNorm2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(3488,3)
        self.Relu = nn.ReLU(inplace=False)
        self.tanH = nn.Tanh()
        self.avgPooling = nn.AdaptiveAvgPool3d((10,4,1))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.Relu(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        x = self.tanH(x)
        
        return(x)
    
    
class rotRegression(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
        channels = 128
    
        self.conv1 = nn.Conv1d(channels,64,kernel_size=1,padding='valid',bias=False)
        self.batchNorm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64,32,kernel_size=1,padding='valid',bias=False)
        self.batchNorm2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(3488,4)
        self.Relu = nn.ReLU(inplace=False)
        self.tanH = nn.Tanh()
        self.avgPooling = nn.AdaptiveAvgPool3d((10,4,1))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.Relu(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        x = self.tanH(x)
        
        return(x)