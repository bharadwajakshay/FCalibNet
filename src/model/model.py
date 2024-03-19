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
from model.crossFeatureHighlight import crossFeatureHighlight

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
        
        # self.featureAnalysis = featureAnalysis()
        self.featureAnalysis = crossFeatureHighlight()
        self.featureMatching = crossFeatureMatching()
        
        self.transRegression = transRegression()
        self.rotRegression = rotRegression()
    


    
    def forward(self,colorImage, lidarImage):
        colorFeatureMap = self.colorEfficientNet(colorImage)
        colorFeatureMap = self.imageUpScaleNet(colorFeatureMap)

        lidarFeatureMap = self.lidarEfficientNet(lidarImage)
        lidarFeatureMap = self.lidarUpScaleNet(lidarFeatureMap)
        
        x = self.featureAnalysis(lidarFeatureMap, colorFeatureMap)
        x = self.featureMatching(x)
        x = x.view(x.shape[0],x.shape[1],-1)
        trans = self.transRegression(x)
        rot = self.rotRegression(x)

        return([rot, trans])
        
