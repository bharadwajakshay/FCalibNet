import torch.nn as nn
from model import efficientNet
from model.crossFeatureMatching import crossFeatureMatching
from model.featureAnalysis import featureAnalysis
from model.regressor import transRegression, rotRegression
import torch
import torchvision
from model.efficientNet import Conv2dNormActivation

class fCalibNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.colorEfficientNet = efficientNet.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT).to('cuda:0')
        self.lidarEfficientNet = efficientNet.efficientnet_v2_lidar_s().to('cuda:0')
        self.colorFeatureMapRed = nn.Sequential(*[Conv2dNormActivation(in_channels=1280, out_channels=1280,kernel_size=3,stride=2,activation_layer=nn.SiLU, norm_layer=nn.BatchNorm2d,padding=(2,1)),
                                                 Conv2dNormActivation(in_channels=1280, out_channels=1280,kernel_size=3,stride=2,activation_layer=nn.SiLU,norm_layer=nn.BatchNorm2d)]).to('cuda:0')
        #self.crossFeatureMatching = crossFeatureMatching()
        
        self.featureAnalysis = featureAnalysis().to('cuda:0')
        self.transRegression = transRegression().to('cuda:1')
        self.rotRegression = rotRegression().to('cuda:1')


    
    def forward(self,colorImage, lidarImage):
        colorFeatureMap = self.colorEfficientNet(colorImage.to('cuda:0'))
        colorFeatureMap = self.colorFeatureMapRed(colorFeatureMap.to('cuda:0'))

        lidarFeatureMap = self.lidarEfficientNet(lidarImage.to('cuda:0'))
        

        reorganizedLiDARFeatureMap = torch.empty_like(colorFeatureMap).unsqueeze(2)
        
        for idx in range(0,lidarFeatureMap.shape[2] - colorFeatureMap.shape[2] ):

            reorganizedLiDARFeatureMap = torch.cat((reorganizedLiDARFeatureMap,
                                        lidarFeatureMap[:,:, idx: idx + colorFeatureMap.shape[2]].unsqueeze(2)), dim=2)
            
        
        reorganizedLiDARFeatureMap = reorganizedLiDARFeatureMap [:,:,1:,:,:]
        
        
        stackedTensor = torch.cat((colorFeatureMap.unsqueeze(2),reorganizedLiDARFeatureMap), dim=2)
        x = self.featureAnalysis(stackedTensor)
        x = x.flatten(start_dim=1)
        
        trans = self.transRegression(x.to('cuda:1'))
        rot = self.rotRegression(x.to('cuda:1'))

        return([rot, trans])
        