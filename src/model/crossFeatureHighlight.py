from model.crossAttention import crossAttention
import torch
import torch.nn as nn

class crossFeatureHighlight(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.attLayer1 = crossAttention()
        self.attLayer2 = crossAttention()
        
    def forward(self, lidar, image):
        lidar = lidar.transpose(-1,-2)
        image = image.transpose(-1,-2)
        x = self.attLayer1 (lidar, image)  
        x = self.attLayer2 (x,image)
        x = torch.concat((x,lidar),dim=1)
        return(x)
