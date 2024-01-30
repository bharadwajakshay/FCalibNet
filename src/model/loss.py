import torch
import torch.nn as nn
import numpy as np
from common import tensorTools, pytorch3D 

class getLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        print("Initializing the loss function")
        
    def forward(self, predTR, colorImg, lidarImg, lidarImgGT, gtTR, projMat):
        
        rot = pytorch3D.quaternion_to_matrix(predTR[0])
        transformationMat =  tensorTools.convSO3NTToSE3(rot,predTR[1])
        invTransformMat = tensorTools.calculateInvRTTensorWhole(transformationMat)
        transformedPoints = tensorTools.applyTransformationOnTensor(lidarImg[:,:3,:,:].transpose(1,3), invTransformMat)
        rangeImg = torch.norm(transformedPoints,2,dim=3)
        idx = torch.where(rangeImg < 1)
        rangeImg = rangeImg.clone()
        rangeImg[rangeImg < 1] = 1e-5
        
        eucledeanDistance = torch.norm(lidarImgGT.transpose(1,3)[:,:,:,:3]-transformedPoints,2,dim=3)
        
        weightImg = torch.div(80, rangeImg)
        weightImg = weightImg.clone()
        weightImg[idx] = 0
        
        weightedEucledeanDist = torch.zeros_like(rangeImg)
        weightedEucledeanDist = eucledeanDistance * weightImg

        
        # Adding Chordal Distance
        # ChordalDistance = ||R1 - R2||F
        chordalDist = torch.empty((transformationMat.shape[0])).cuda()
        euclideanDist = torch.empty((transformationMat.shape[0])).cuda()
        for batch in range(transformationMat.shape[0]):
            chordalDist[batch] = torch.linalg.matrix_norm(transformationMat[batch,:3,:3] - gtTR[batch,:3,:3],'fro')
            if torch.isnan(chordalDist[batch]):
                print('Breakpoint')
            euclideanDist[batch] = weightedEucledeanDist[batch,:,:].mean()
        
        eucledianMatDist = torch.linalg.norm(transformationMat[:,:3,3] - gtTR[:,:3,3],2,dim=1)
        
        totalLoss = torch.mean(chordalDist) + torch.mean(eucledianMatDist) + torch.mean(euclideanDist)
        
        return(totalLoss)
        
        