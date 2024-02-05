import torch
import torch.nn as nn
import numpy as np
from common import tensorTools, pytorch3D 

from external.Chamfer3D.dist_chamfer_3D import chamfer_3DDist

import config

chamfer_dist = chamfer_3DDist()

class getLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.loss = config.loss
        self.lossWeight = config.lossWeight
        
        print("Initializing the loss function")
        
    def forward(self, predTR, colorImg, lidarImg, lidarImgGT, gtTR, projMat):
        
        lidarImg = lidarImg.to(predTR[0].device)
        lidarImgGT = lidarImgGT.to(predTR[0].device)
        gtTR = gtTR.to(predTR[0].device)
        
        rot = pytorch3D.quaternion_to_matrix(predTR[0])
        transformationMat =  tensorTools.convSO3NTToSE3(rot,predTR[1])
        invTransformMat = tensorTools.calculateInvRTTensorWhole(transformationMat)
        transformedPoints = tensorTools.applyTransformationOnTensor(lidarImg[:,:3,:,:].transpose(1,3), invTransformMat)
        
        
        if ("WEUC" in self.loss) or ("EUC" in self.loss):
            # get Eucledian distance
            eucledeanDistance = torch.norm(lidarImgGT.transpose(1,3)[:,:,:,:3]-transformedPoints,2,dim=3)

            if("WEUC" in self.loss):
                # get Weighted Euclidian distance
                rangeImg = torch.norm(transformedPoints,2,dim=3)
                idx = torch.where(rangeImg < 1)
                rangeImg = rangeImg.clone()
                rangeImg[rangeImg < 1] = 1e-5
        
                weightImg = torch.div(80, rangeImg)
                weightImg = weightImg.clone()
                weightImg[idx] = 0
                weightedEucledeanDist = eucledeanDistance * weightImg

                mean = weightedEucledeanDist.view(weightedEucledeanDist.shape[0],-1).mean(-1,keepdim=True).mean()

                # get weight of the loss function
                lossweight = self.lossWeight[self.loss.index("WEUC")]

                pointloss = mean*lossweight
            else:
                mean = eucledeanDistance.view(eucledeanDistance.shape[0],-1).mean(-1,keepdim=True).mean()
                lossweight = self.lossWeight[self.loss.index("EUC")]
                pointloss = mean*lossweight

        if ("CHAMP" in self.loss):
            # Get Champer distance'
            d1, d2, _, _ = chamfer_dist(transformedPoints.reshape((transformedPoints.shape[0],-1,3)),
                                        lidarImgGT.transpose(1,3)[:,:,:,:3].reshape((lidarImgGT.shape[0],-1, 3)))
            d1 = torch.mean(torch.sqrt(d1))
            d2 = torch.mean(torch.sqrt(d2))
            d = (d1 + d2) / 2
            lossweight = self.lossWeight[self.loss.index("CHAMP")]
            pointloss = d * lossweight
        else:
            pointloss = 0


        if ("CHORDAL" in self.loss):
            # Adding Chordal Distance
            # ChordalDistance = ||R1 - R2||F
            chordalDist = torch.empty((transformationMat.shape[0])).cuda()
            for batch in range(transformationMat.shape[0]):
                chordalDist[batch] = torch.linalg.matrix_norm(transformationMat[batch,:3,:3] - gtTR[batch,:3,:3],'fro')
            
            lossweight = self.lossWeight[self.loss.index("CHORDAL")]

            chordaLoss = lossweight * chordalDist.mean()
        else:
            chordaLoss = 0

        if ("EUCTR" in self.loss):
            eucledianMatDist = torch.linalg.norm(transformationMat[:,:3,3] - gtTR[:,:3,3],2,dim=1)
            lossweight = self.lossWeight[self.loss.index("EUCTR")]
            transfomTranslationLoss = lossweight * eucledianMatDist.mean()
        else:
            transfomTranslationLoss = 0


        totalLoss = pointloss + chordaLoss + transfomTranslationLoss
        
        return(totalLoss)
        
        