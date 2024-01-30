import torch
import torch.nn as nn


class featureAnalysis(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        channels = 1408
        kernelSize = 3
        padding = (1,1)
        stride = (1,1)
        
        self.conv1x1B0 = nn.Conv2d(channels, 1024, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B0.weight)
        
        self.conv1x1B1 = nn.Conv2d(1024, 1024, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B1.weight)
        
        self.conv1x1B2 = nn.Conv2d(1024, 512, kernelSize, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv1x1B2.weight)
        
        # Batch Normalization 
        self.bn0 = nn.BatchNorm2d(1024)
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)

        
        self.Relu = nn.ReLU(inplace=False)
        
    def forward(self,x):
        
        x = self.conv1x1B0(x)
        x = self.bn0(x)
        x = self.Relu(x)
        
        x = self.conv1x1B1(x)
        x = self.bn1(x)
        x = self.Relu(x)
        
        x = self.conv1x1B2(x)
        x = self.bn2(x) 
        x = self.Relu(x)
        
        return(x)
        