import torch

import numpy as np
from tqdm import tqdm
import os
import datetime
from model.model import fCalibNet
from dataLoader import dataLoader
from datetime import datetime
from model.loss import getLoss
from common.tensorTools import saveCheckPoint

from fcalibValidation import valitation

import config

torch.autograd.set_detect_anomaly(True)

def main():

    if not os.path.exists(config.checkpointDir):
        os.makedirs(config.checkpointDir)

    if not os.path.exists(config.logDir):
        os.makedirs(config.logDir)
    
    currentTimeStamp = datetime.now()

    torch.cuda.empty_cache()

    #define your model here 
    model = fCalibNet()

    if torch.cuda.is_available():
        device = 'cuda'
        if config.DDPT:
            #TODO: enable Distributed data parallel training
            print('Placeholder as I need to something for syntax')
        elif torch.cuda.device_count()>1 and config.DPT:
            print('Multiple GPUs found. Moving to Dataparallel approach')
            model = torch.nn.DataParallel(model)
        else:
            print('Training on a single GPU')
    else:
        device = 'cpu'

    model = model.to(device)
    loss = getLoss().to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
                params = model.parameters(),
                lr = config.training['learningRate'],
                betas = (config.training['beta0'], config.training['beta1']),
                weight_decay = config.training['decayRate']
                )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', patience= 3)


    trainingData = dataLoader(config.datasetFile)
    validationData = dataLoader(config.datasetFile, 'test') 
    trainingDataLoader = torch.utils.data.DataLoader(trainingData,
                                                     batch_size=config.training['batchSize'],
                                                     shuffle=True,
                                                     num_workers=14,
                                                     drop_last=True)
    
    validatingDataLoader = torch.utils.data.DataLoader(validationData,
                                                     batch_size=config.training['batchSize'],
                                                     shuffle=True,
                                                     num_workers=14,
                                                     drop_last=True)

    monitorDist = 10000000
    for epochs in range(0,config.training['epoch']):
        # put model in training mode
        model = model.train()
        # Freeze the colormodel weights as pretrained weights are being used
        model.colorEfficientNet = model.colorEfficientNet.eval()
        for params in model.colorEfficientNet.parameters():
            params.requires_grad = False


        #  get your data
        lossValueArray = torch.zeros(len(trainingDataLoader))
        for dataBatch, data in tqdm(enumerate(trainingDataLoader,0),total=len(trainingDataLoader)):
            optimizer.zero_grad()
            
            colorImage, lidarImage, gtLidarImage, gtTransfrom, projectionMat = data
             
            colorImage = torch.transpose(colorImage.to(device),1,3)
            lidarImage = torch.transpose(lidarImage.to(device),1,3).type(torch.cuda.FloatTensor)
            gtLidarImage = torch.transpose(gtLidarImage.to(device),1,3).type(torch.cuda.FloatTensor)
            projectionMat = projectionMat.to(device)
            gtTransfrom = gtTransfrom.to(device)
            modelTime = datetime.now()
            [predRot, predTrans] = model(colorImage, lidarImage)
            modelTimeEnd = datetime.now()
            lossVal = loss([predRot, predTrans], colorImage, lidarImage, gtLidarImage, gtTransfrom, projectionMat)
            lossTimeEnd = datetime.now()
            lossVal.backward()
            backwardTimeEnd = datetime.now()
            
            #print(f"Model Execution time = {(modelTimeEnd.second + modelTimeEnd.microsecond*1e-6) - (modelTime.second + modelTime.microsecond*1e-6)}sec\t \
            #          Loss Execution time= {(lossTimeEnd.second + lossTimeEnd.microsecond*1e-6 ) - (modelTimeEnd.second + modelTimeEnd.microsecond*1e-6)}sec\t \
            #          Backward Execution time = {(backwardTimeEnd.second + backwardTimeEnd.microsecond*1e-6 ) - (lossTimeEnd.second + lossTimeEnd.microsecond*1e-6)})")
            
            optimizer.step()
            
            lossValueArray[dataBatch] = lossVal

        
        
        # Validation
        with torch.no_grad():
            model = model.eval()
            SE3Dist, eucledianDist, translation, eulerAngle = valitation(model, validatingDataLoader, device)
            
        # Scheduler step
        scheduler.step(np.mean(eucledianDist))
        print(f"Mean loss value = {lossVal}")
        print(f"Epoch: {epochs}\t Mean SE3 Dist = {np.mean(SE3Dist)} \t Mean Eucledian Distance = {np.mean(eucledianDist)}")
        
        if monitorDist > np.mean(eucledianDist):
            monitorDist = np.mean(eucledianDist)
            if not os.path.exists(config.checkpointDir):
                os.makedirs(config.checkpointDir)
            timeStamp = datetime.now()
            checkpointPath = os.path.join(config.checkpointDir,
                                          '_'.join([str(timeStamp.year),
                                                    str(timeStamp.month), 
                                                    str(timeStamp.day), 
                                                    ":".join([str(timeStamp.hour),str(timeStamp.minute)]),
                                                    f"ED_{np.mean(SE3Dist)}"])+'.pth')
            saveCheckPoint(model, optimizer, epochs, lossVal, scheduler, checkpointPath)
            




if __name__ == '__main__':
    main()
