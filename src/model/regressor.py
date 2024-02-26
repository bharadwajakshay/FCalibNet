import torch.nn as nn
import torch

class transRegression(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
        channels = 128
    
        self.conv1 = nn.Conv3d(channels,64,kernel_size=3,padding='valid',bias=False)
        self.batchNorm1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64,32,kernel_size=3,padding='valid',bias=False)
        self.batchNorm2 = nn.BatchNorm3d(32)
        self.fc = nn.Linear(1280,3)
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
        x = self.avgPooling(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        
        return(x)
    
    
class rotRegression(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
        channels = 128
    
        self.conv1 = nn.Conv3d(channels,64,kernel_size=3,padding='valid',bias=False)
        self.batchNorm1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64,32,kernel_size=3,padding='valid',bias=False)
        self.batchNorm2 = nn.BatchNorm3d(32)
        self.fc = nn.Linear(1280,4)
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
        x = self.avgPooling(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        x = self.tanH(x)
        
        return(x)