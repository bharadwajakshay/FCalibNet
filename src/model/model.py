import torch.nn as nn
from model import efficientNet
from model.crossFeatureMatching import crossFeatureMatching
from model.featureAnalysis import featureAnalysis
from model.regressor import transRegression, rotRegression
import torch
import torchvision
from model.efficientNet import Conv2dNormActivation
from collections import OrderedDict
from torchinfo import summary

class fCalibNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.colorEfficientNet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.colorEfficientNet = torch.nn.Sequential(OrderedDict([*(list(self.colorEfficientNet.named_children())[:-2])]))
        
        self.lidarEfficientNet = torchvision.models.resnet50()
        # Replace the 1st layer to accept 4 channels
        self.lidarEfficientNet.conv1 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

        self.lidarEfficientNet = torch.nn.Sequential(OrderedDict([*(list(self.lidarEfficientNet.named_children())[:-2])]))
        
        #TODO: This operation takes a lot of memory. This can be replaced by a 1x1 kernel.. Check if that works later
        self.lidarUpScaleNet = nn.Sequential(*[nn.ConvTranspose2d(2048,2048,kernel_size=(3,3),stride=(2,2)),nn.ConvTranspose2d(2048,1024,kernel_size=(3,3),stride=(2,2))])
        self.imageUpScaleNet = nn.Sequential(*[nn.ConvTranspose2d(2048,2048,kernel_size=(3,3),stride=(1,1)),nn.ConvTranspose2d(2048,1024,kernel_size=(3,3),stride=(1,1))])
        
        self.featureAnalysis = featureAnalysis()
        self.transRegression = transRegression()
        self.rotRegression = rotRegression()
        self.adaptiveAvgPool = nn.AdaptiveAvgPool3d((14,1,1))


    
    def forward(self,colorImage, lidarImage):
        colorFeatureMap = self.colorEfficientNet(colorImage)
        colorFeatureMap = self.imageUpScaleNet(colorFeatureMap)

        lidarFeatureMap = self.lidarEfficientNet(lidarImage)
        lidarFeatureMap = self.lidarUpScaleNet(lidarFeatureMap)
        

        reorganizedLiDARFeatureMap = torch.empty_like(colorFeatureMap).unsqueeze(2)
        
        for idx in range(colorFeatureMap.shape[2],lidarFeatureMap.shape[2], 7):

            reorganizedLiDARFeatureMap = torch.cat((reorganizedLiDARFeatureMap,
                                        lidarFeatureMap[:,:, idx - colorFeatureMap.shape[2]: idx].unsqueeze(2)), dim=2)
        
        #reorganizedLiDARFeatureMap = torch.cat((reorganizedLiDARFeatureMap,
        #                                lidarFeatureMap[:,:, (lidarFeatureMap.shape[2]-colorFeatureMap.shape[2])-1: -1].unsqueeze(2)), dim=2) 
        
        reorganizedLiDARFeatureMap = reorganizedLiDARFeatureMap [:,:,1:,:,:]
        
        
        stackedTensor = torch.cat((colorFeatureMap.unsqueeze(2),reorganizedLiDARFeatureMap), dim=2)
        x = self.featureAnalysis(stackedTensor)
        
        trans = self.transRegression(x)
        rot = self.rotRegression(x)

        return([rot, trans])
        