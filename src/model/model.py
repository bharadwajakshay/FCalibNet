import torch.nn as nn
from model import efficientNet
from model.crossFeatureMatching import crossFeatureMatching
from model.featureAnalysis import featureAnalysis
from model.regressor import transRegression, rotRegression
import torch
import torchvision

class fCalibNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.colorEfficientNet = efficientNet.efficientnet_v2_camera_s()
        self.lidarEfficientNet = efficientNet.efficientnet_v2_lidar_s()
        self.crossFeatureMatching = crossFeatureMatching()
        self.featureAnalysis = featureAnalysis()
        self.transRegression = transRegression()
        self.rotRegression = rotRegression()


    
    def forward(self,colorImage, lidarImage):
        colorFeatureMap = self.colorEfficientNet(colorImage)
        lidarFeatureMap = self.lidarEfficientNet(lidarImage)

        reorganizedLiDARFeatureMap = torch.empty_like(colorFeatureMap).unsqueeze(2)
        
        for idx in range(0,lidarFeatureMap.shape[2] - colorFeatureMap.shape[2] ):

            reorganizedLiDARFeatureMap = torch.cat((reorganizedLiDARFeatureMap,
                                        lidarFeatureMap[:,:, idx: idx + colorFeatureMap.shape[2]].unsqueeze(2)), dim=2)
            
        
        reorganizedLiDARFeatureMap = reorganizedLiDARFeatureMap [:,:,1:,:,:]
        
        compTensorStack = torch.empty((colorFeatureMap.shape[0],64, colorFeatureMap.shape[2], colorFeatureMap.shape[3])).unsqueeze(1).cuda()
        
        for slice in range(0, reorganizedLiDARFeatureMap.shape[2]):
            compTensor = torch.cat((colorFeatureMap,reorganizedLiDARFeatureMap[:,:,slice,:,:].squeeze(2)),dim=1)
            x = self.crossFeatureMatching(compTensor)
            compTensorStack = torch.cat((compTensorStack,x.unsqueeze(1)), dim=1)
            
            
        compTensorStack = compTensorStack[:,1:,:,:,:]
        compTensorStackResized = compTensorStack.reshape((compTensorStack.shape[0], compTensorStack.shape[1]*compTensorStack.shape[2],compTensorStack.shape[3],compTensorStack.shape[4]))
        
        x = self.featureAnalysis(compTensorStackResized)
        x = x.flatten(start_dim=1)
        
        trans = self.transRegression(x)
        rot = self.rotRegression(x)

        return([rot, trans])
        