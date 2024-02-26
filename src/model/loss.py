import torch
import torch.nn as nn
import numpy as np
from common import tensorTools, pytorch3D 

from external.Chamfer3D.dist_chamfer_3D import chamfer_3DDist

import config
import logging

chamfer_dist = chamfer_3DDist()

class getLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.loss = config.loss
        self.lossWeight = config.lossWeight
        
        logging.info(f"Initializing the loss function. Loss details:\n \
                     \t Loss Options:[{config.loss}]\n \
                     \t Loss Weight: [{config.lossWeight}]")
        
    def forward(self, predTR, colorImg, lidarImg, lidarImgGT, gtTR, projMat):
        
        lidarImg = lidarImg.to(predTR[0].device)
        lidarImgGT = lidarImgGT.to(predTR[0].device)
        gtTR = gtTR.to(predTR[0].device)
        
        rot = pytorch3D.quaternion_to_matrix(predTR[0])
        transformationMat =  tensorTools.convSO3NTToSE3(rot,predTR[1])
        invTransformMat = tensorTools.calculateInvRTTensorWhole(transformationMat)
        transformedPoints = tensorTools.applyTransformationOnTensor(lidarImg[:,:3,:,:].transpose(1,3), invTransformMat)
        
        WEUCLoss = torch.tensor(0)
        EUCLoss = torch.tensor(0)
        CHAMPLoss = torch.tensor(0)
        EMDLoss = torch.tensor(0)
        MANLoss = torch.tensor(0)
        
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

                WEUCLoss = mean*lossweight
                
            elif ("EUC" in self.loss):
                mean = eucledeanDistance.view(eucledeanDistance.shape[0],-1).mean(-1,keepdim=True).mean()
                lossweight = self.lossWeight[self.loss.index("EUC")]
                EUCLoss = mean*lossweight

        if ("MANHATTAN" in self.loss):
            manhattanDistance = torch.norm(lidarImgGT.transpose(1,3)[:,:,:,:3]-transformedPoints,1,dim=3)
            mean = manhattanDistance.view(manhattanDistance.shape[0],-1).mean(-1,keepdim=True).mean()
            lossweight = self.lossWeight[self.loss.index("MANHATTAN")]
            MANLoss = mean * lossweight

        if ("CHAMP" in self.loss):
            # Get Champer distance'
            d1, d2, _, _ = chamfer_dist(transformedPoints.reshape((transformedPoints.shape[0],-1,3)),
                                        lidarImgGT.transpose(1,3)[:,:,:,:3].reshape((lidarImgGT.shape[0],-1, 3)))
            d1 = torch.mean(torch.sqrt(d1))
            d2 = torch.mean(torch.sqrt(d2))
            d = (d1 + d2) / 2
            lossweight = self.lossWeight[self.loss.index("CHAMP")]
            CHAMPLoss = d * lossweight
        else:
            CHAMPLoss = torch.tensor(0)


        if ("CHORDAL" in self.loss):
            # Adding Chordal Distance
            # ChordalDistance = ||R1 - R2||F
            chordalDist = torch.empty((transformationMat.shape[0])).cuda()
            for batch in range(transformationMat.shape[0]):
                chordalDist[batch] = torch.linalg.matrix_norm(transformationMat[batch,:3,:3] - gtTR[batch,:3,:3],'fro')
            
            lossweight = self.lossWeight[self.loss.index("CHORDAL")]

            chordaLoss = lossweight * chordalDist.mean()
        else:
            chordaLoss = torch.tensor(0)

        if ("EUCTR" in self.loss):
            eucledianMatDist = torch.linalg.norm(transformationMat[:,:3,3] - gtTR[:,:3,3],2,dim=1)
            lossweight = self.lossWeight[self.loss.index("EUCTR")]
            transfomTranslationLoss = lossweight * eucledianMatDist.mean()
        else:
            transfomTranslationLoss = torch.tensor(0)


        totalLoss = WEUCLoss.to(rot.device) + EMDLoss.to(rot.device) +\
                    CHAMPLoss.to(rot.device) + EUCLoss.to(rot.device) +\
                    chordaLoss.to(rot.device) + transfomTranslationLoss.to(rot.device) +\
                    MANLoss.to(rot.device)
        
        return(totalLoss)
        
        