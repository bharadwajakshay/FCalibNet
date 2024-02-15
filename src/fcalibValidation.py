import torch
from common import tensorTools
from scipy.spatial.transform import Rotation as R

import numpy as np
from tqdm import tqdm
import config

def valitation(model, dataLoader, device):
    simpleDistanceSE3 = np.empty((len(dataLoader)*config.training['batchSize'],3))
    eucledianDistance = np.empty((len(dataLoader)*config.training['batchSize'],3))
    translation = np.empty((len(dataLoader)*config.training['batchSize'],3))
    eulerAngle = np.empty((len(dataLoader)*config.training['batchSize'],3))
    translationGT = np.empty((len(dataLoader)*config.training['batchSize'],3))
    eulerAngleGT = np.empty((len(dataLoader)*config.training['batchSize'],3))
    translationErr = np.empty((len(dataLoader)*config.training['batchSize'],3))
    eulerAngleERR = np.empty((len(dataLoader)*config.training['batchSize'],3))
    
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
        
        RtR = torch.bmm(tensorTools.calculateInvRTTensorWhole(predictedTR),gtTransfrom.to(predictedTR.device))
        I = torch.eye(RtR.shape[1]).unsqueeze(0).to(RtR.device)
        I = I.repeat((RtR.shape[0],1,1))
        
        # Eucledian distance
        invTransformMat = tensorTools.calculateInvRTTensorWhole(predictedTR)
        transformedPoints = tensorTools.applyTransformationOnTensor(lidarImage[:,:3,:,:].transpose(1,3), invTransformMat)
        eucledeanDist = torch.norm(gtLidarImage.transpose(1,3)[:,:,:,:3]-transformedPoints.to(gtLidarImage.device),2,dim=3)
        
        # get angular error 
        

        for idx in range(0, RtR.shape[0]):
            globalIdx = dataBatch*RtR.shape[0] + idx
            simpleDistanceSE3[globalIdx] = torch.norm((RtR[idx,:,:] - I[idx,:,:]),'fro').cpu().detach().numpy()
            
            # Eucledean Distance
            eucledianDistance[globalIdx] =  torch.mean(eucledeanDist[idx,:,:]).detach().cpu().numpy()
            
            # Rotation
            rPred = R.from_matrix(predictedTR[idx,:3,:3].cpu().detach().numpy())
            eulerAngle[globalIdx,:] = rPred.as_euler('zxy', degrees=True)

            # Get GT rotation
            rGT = R.from_matrix(gtTransfrom[idx,:3,:3].cpu().detach().numpy())
            eulerAngleGT[globalIdx,:] = rGT.as_euler('zxy', degrees=True)

            # Translation
            translation[globalIdx,:] = predictedTR[idx,:3,3].cpu().detach().numpy()

            # GT Translation
            translationGT[globalIdx,:] = gtTransfrom[idx,:3,3].cpu().detach().numpy()
            

        
    absoluteAngularError = np.abs(eulerAngleGT - eulerAngle)
    absoluteTranslationError = np.abs(translationGT - translation)
        
    return(simpleDistanceSE3,eucledianDistance, translation, eulerAngle, absoluteAngularError, absoluteTranslationError)    
        