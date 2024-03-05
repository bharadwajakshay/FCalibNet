import torch
import torch.nn as nn

class crossFeatureMatching(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        channels = 2048
        kernelSize = 3
        padding = 'valid'
        stride = (1,1)
        self.conv1x1B0 = nn.Conv2d(channels, 2048, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B0.weight)
        self.conv1x1B1 = nn.Conv2d(2048, 1024,kernelSize, padding=padding, stride=stride, bias=False)
        nn.init.xavier_uniform_(self.conv1x1B1.weight)
        self.conv1x1B2 = nn.Conv2d(1024, 512,kernelSize, padding=padding, stride=stride, bias=False)
        nn.init.xavier_uniform_(self.conv1x1B2.weight)
        self.conv1x1B3 = nn.Conv2d(512, 256,kernelSize, padding=padding, stride=stride, bias=False)
        nn.init.xavier_uniform_(self.conv1x1B3.weight)
        self.conv1x1B4 = nn.Conv2d(256, 128,kernelSize, padding=padding, stride=stride, bias=False)
        nn.init.xavier_uniform_(self.conv1x1B4.weight)
        
        # Batch Normalization 
        self.bn0 = nn.BatchNorm2d(2048)
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.Relu = nn.ReLU(inplace=False)
        
    def forward(self, x): 
        
        x = self.conv1x1B0(x)
        x = self.bn0(x)
        x = self.Relu(x)
        x = self.conv1x1B1(x)
        x = self.bn1(x)
        x = self.Relu(x)
        x = self.conv1x1B2(x)
        x = self.bn2(x)
        x = self.Relu(x)
        x = self.conv1x1B3(x)
        x = self.bn3(x)
        x = self.Relu(x)
        x = self.conv1x1B4(x)
        x = self.bn4(x)
        x = self.Relu(x)
        
        return x
    
    