import torch.nn as nn


class transRegression(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
        channels = 20608
    
        self.linerlayer1 = nn.Linear(channels, int(channels/2))
        self.linerlayer2 = nn.Linear(int(channels/2), int(channels/4))
        self.linerlayer3 = nn.Linear(int(channels/4), 3)
        
        self.Relu = nn.ReLU(inplace=False)
        self.tanH = nn.Tanh()
        
    def forward(self, x):
        x = self.linerlayer1(x)
        x = self.tanH(x)
        
        x = self.linerlayer2(x)
        x = self.tanH(x)
        
        x = self.linerlayer3(x)
        x = self.tanH(x)
        
        return(x)
    
    
class rotRegression(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
        channels = 20608
    
        self.linerlayer1 = nn.Linear(channels, int(channels/2))
        self.linerlayer2 = nn.Linear(int(channels/2), int(channels/4))
        self.linerlayer3 = nn.Linear(int(channels/4), 4)
        
        self.Relu = nn.ReLU(inplace=False)
        self.tanH = nn.Tanh()
        
    def forward(self, x):
        x = self.linerlayer1(x)
        x = self.tanH(x)
        
        x = self.linerlayer2(x)
        x = self.tanH(x)
        
        x = self.linerlayer3(x)
        x = self.tanH(x)
        
        return(x)