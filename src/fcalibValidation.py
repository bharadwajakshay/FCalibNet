import torch
from common import tensorTools
from scipy.spatial.transform import Rotation as R

import numpy as np
from tqdm import tqdm
import config

def valitation(model, dataLoader, device):
    simpleDistanceSE3 = np.empty(len(dataLoader))
    eucledianDistance = np.empty(len(dataLoader))
    translation = np.empty((len(dataLoader)*config.training['batchSize'],3))
    eulerAngle = np.empty((len(dataLoader)*config.training['batchSize'],3))
    
    
    for dataBatch, data in tqdm(enumerate(dataLoader,0),total=len(dataLoader)):
        
        colorImage, lidarImage, gtLidarImage, gtTransfrom, projectionMat = data
        colorImage = torch.transpose(colorImage.to(device),1,3)
        lidarImage = torch.transpose(lidarImage.to(device),1,3).type(torch.cuda.FloatTensor)
        gtLidarImage = torch.transpose(gtLidarImage.to(device),1,3).type(torch.cuda.FloatTensor)
        projectionMat = projectionMat.to(device).type(torch.cuda.FloatTensor)
        gtTransfrom = gtTransfrom.to(device).type(torch.cuda.FloatTensor)
        [predRot, predTrans] = model(colorImage, lidarImage)
        
        predRotMat = tensorTools.quaternion_to_matrix(predRot)
        
        predictedTR = tensorTools.convSO3NTToSE3(predRotMat,predTrans)
        
        RtR = torch.bmm(tensorTools.calculateInvRTTensorWhole(predictedTR),gtTransfrom)
        I = torch.eye(RtR.shape[1]).unsqueeze(0).to(device)
        I = I.repeat((RtR.shape[0],1,1))
        
        # Eucledian distance
        invTransformMat = tensorTools.calculateInvRTTensorWhole(predictedTR)
        transformedPoints = tensorTools.applyTransformationOnTensor(lidarImage[:,:3,:,:].transpose(1,3), invTransformMat)
        eucledeanDist = torch.norm(gtLidarImage.transpose(1,3)[:,:,:,:3]-transformedPoints,2,dim=3)
        
        SE3Dist = np.empty(RtR.shape[0])
        eucledianDistanceBatch = np.empty(RtR.shape[0])
        for idx in range(0,RtR.shape[0]):
            SE3Dist[idx] = torch.norm((RtR[idx,:,:] - I[idx,:,:]),'fro').cpu().detach().numpy()
            
            #Rotation
            rPred = R.from_matrix(predictedTR[idx,:3,:3].cpu().detach().numpy())
            eulerAngle[idx,:] = rPred.as_euler('zxy', degrees=True)  
            translation[idx,:] = predictedTR[idx,:3,3].cpu().detach().numpy()
            
            # Eucledean Distance
            eucledianDistanceBatch[idx] =  torch.mean(eucledeanDist[idx,:,:]).detach().cpu().numpy()
            
        
        simpleDistanceSE3[dataBatch] = np.mean(SE3Dist)
        eucledianDistance[dataBatch] = np.mean(eucledianDistanceBatch)
        
        
        
    return(SE3Dist,eucledianDistance, translation, eulerAngle )    
        