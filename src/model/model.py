import torch.nn as nn
from model import efficientNet

class fCalibNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.colorEfficientNet = efficientNet.efficientnet_v2_camera_s()
        self.lidarEfficientNet = efficientNet.efficientnet_v2_lidar_s()


    
    def forward(self,colorImage, lidarImage):
        colorFeatureMap = self.colorEfficientNet(colorImage)
        lidarFeatureMap = self.lidarEfficientNet(lidarImage)
        print('placeholder')
        