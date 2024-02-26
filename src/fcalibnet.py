import torch
import time
import numpy as np
from tqdm import tqdm
import os
import datetime
from model.model import fCalibNet
from dataLoader import dataLoader
from datetime import datetime
from common.tensorTools import saveCheckPoint

from fcalibValidation import valitation

import config
import logging

# Get current system
programStartTime = datetime.now()

# Enabling logging
logfileDir = os.path.join(config.logDir, config.mode)
if not os.path.exists(logfileDir):
    os.makedirs(logfileDir)
logfile = os.path.join(logfileDir,'-'.join((config.name,str(programStartTime.date()),
                                    str(programStartTime.hour),
                                    str(programStartTime.minute)))+'.log')

logging.basicConfig(filename=logfile,
                    filemode='w',
                    format='%(asctime)s:%(name)s - %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

torch.autograd.set_detect_anomaly(True)

from model.loss import getLoss

def main():
    
    logging.info(f"Training network {config.name}.")

    if not os.path.exists(config.checkpointDir):
        logging.info("Checkpoint directory not found. Creating checkpoint directory.")
        os.makedirs(config.checkpointDir)

    torch.cuda.empty_cache()

    #define your model here 
    model = fCalibNet()

    if torch.cuda.is_available():
        device = 'cuda'
        if config.DDPT:
            #TODO: enable Distributed data parallel training
            logging.info('Placeholder as I need to something for syntax')
        elif torch.cuda.device_count()>1 and config.DPT:
            logging.info('Multiple GPUs found. Moving to Dataparallel approach')
            model = torch.nn.DataParallel(model)
        else:
            logging.info('Training on a single GPU')
            
        if config.modelParallel:
            if not torch.cuda.device_count()>1:
                logging.error("Only a single GPU is found. Cant train with model parallel mode. Exiting")
                exit(-1)
            else:
                logging.info(f"{torch.cuda.device_count()} GPUS found. Enabling Model parallel training")
    else:
        logging.info("Using default CPU for training")
        device = 'cpu'

    model = model.to(device) #ASB handeled in the model constructor
    loss = getLoss().to(device)
    lastepoch = -1
    
    # Setup optimizer
    if config.optimizer == 'Adam':
        logging.info(f"Setting up optimizer.\n Adam Optimizer:\n Parameters:\n \
                \t Learning Rate: {config.training['learningRate']}\n \
                \t Betas: {config.training['beta0']} - {config.training['beta1']}\n \
                \t Weight Decay: {config.training['decayRate']}")
        
        optimizer = torch.optim.Adam(
            params = model.parameters(),
            lr = config.training['learningRate'],
            betas = (config.training['beta0'], config.training['beta1']),
            weight_decay = config.training['decayRate'])
        
    elif config.optimizer == 'SGD':
        logging.info(f"Setting up optimizer. Selecting SGD Optimizer:\n  Parameters:\n \
                \t Learning Rate: {config.training['learningRate']}\n \
                \t Betas: {config.training['beta0']} - {config.training['beta1']}\n \
                \t Weight Decay: {config.training['decayRate']}\n \
                \t Momentum: {config.training['momentum']}")
        
        optimizer = torch.optim.SGD(
            params = model.parameters(),
            lr = config.training['learningRate'],
            weight_decay = config.training['decayRate'],
            momentum = config.training['momentum'])
    
    else:
        logging.error("Current optimizer setting is not supported. Exititng.")
        exit(-1)

    if config.scheduler['name'] == 'ROP':
        logging.info(f"Setting up scheduler. Selected scheduler is Reduce LR on Plateau.\n \
                     Scheduler parameters: \n \
                     \t Mode : {config.scheduler['mode']} \n \
                     \t Patience : {config.scheduler['patience']}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=config.scheduler['mode'],
                                                               patience= config.scheduler['patience'])
    elif config.scheduler['name'] == 'MLR':
        logging.info(f"Setting up scheduler. Selected scheduler is Multistep LR.\n \
                     Scheduler parameters: \n \
                     \t Steps : {config.scheduler['step']}")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        last_epoch=lastepoch,
                                                        milestones=config.scheduler['step'])

    else:
        logging.error("Current scheduler setting is not supported. Exititng.")
        exit(-1)

    # Load the checkpoint
    logging.info("Checking if training needs to resume from checkpoint")
    loadedEpochs = 0
    if config.training['loadFromCheckpoint']:
        logging.info("Resuming from checkpoint is enabled. Trying to load checkpoint.")
        if os.path.exists(os.path.join(config.checkpointDir,config.chkPointFileName)):
            logging.info("Found Checkpoint.. Loading the model weights")
            modelWeights = torch.load(os.path.join(config.checkpointDir,config.chkPointFileName))
            logging.debug(f"Successfully loaded the {os.path.join(config.checkpointDir,config.chkPointFileName)} file ")
            loadedEpochs = modelWeights['epoch']
            logging.debug(f"Successfully loaded previous training epoch :{loadedEpochs}")
            optimizer.load_state_dict(modelWeights['optimizer_state_dict'])
            logging.debug(f"Successfully loaded optimizer parameters")
            scheduler.load_state_dict(modelWeights['scheduler_state_dict'])
            logging.debug(f"Successfully loaded scheduler parameters")
            model.load_state_dict(modelWeights['model_state_dict'])
            logging.debug(f"Successfully loaded mode weights")
        else:
            logging.error("Unable to find checkpoint. Proceeding without loading the weights. Training from epoch 0 ")
    else:
        logging.info("Resuming from checkpoint is Disabled. Training from scratch.")


    logging.info(f"Loading dataloader. Datafile: {config.datasetFile}")
    trainingData = dataLoader(config.datasetFile)
    validationData = dataLoader(config.datasetFile, 'test') 
    trainingDataLoader = torch.utils.data.DataLoader(trainingData,
                                                     batch_size=config.training['batchSize'],
                                                     shuffle=True,
                                                     num_workers=10,
                                                     drop_last=True)
    
    validatingDataLoader = torch.utils.data.DataLoader(validationData,
                                                     batch_size=config.training['batchSize'],
                                                     shuffle=True,
                                                     num_workers=10,
                                                     drop_last=True)

    monitorDist = 10000000
    previousfilePath = None
    logging.info("Starting to train the network")
    
    
    # Start profiling
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True)
    
    for epochs in range(loadedEpochs,config.training['epoch']):
        epochStartTime = time.time()
        logging.info(f"Training details:\n \
                     \t Epoch: {epochs}\n \
                     \t Batch Size: {config.training['batchSize']}\n \
                     \t Learning Rate:{optimizer.param_groups[0]['lr']}")

        
        # put model in training mode
        model = model.train()
        # Freeze the colormodel weights as pretrained weights are being used
        model.colorEfficientNet = model.colorEfficientNet.eval()
        for params in model.colorEfficientNet.parameters():
            params.requires_grad = False

        prof.start()
        #  get your data
        lossValueArray = torch.zeros(len(trainingDataLoader))
        for dataBatch, data in tqdm(enumerate(trainingDataLoader,0),total=len(trainingDataLoader)):
            optimizer.zero_grad()
            
            colorImage, lidarImage, gtLidarImage, gtTransfrom, projectionMat = data
             
            colorImage = torch.transpose(colorImage.to(device),1,3)
            lidarImage = torch.transpose(lidarImage.to(device),1,3).type(torch.cuda.FloatTensor)
            gtLidarImage = torch.transpose(gtLidarImage.to(device),1,3).type(torch.cuda.FloatTensor)
            projectionMat = projectionMat.to(device).type(torch.cuda.FloatTensor)
            gtTransfrom = gtTransfrom.to(device).type(torch.cuda.FloatTensor)

            [predRot, predTrans] = model(colorImage, lidarImage)

            lossVal = loss([predRot, predTrans], colorImage, lidarImage, gtLidarImage, gtTransfrom, projectionMat)

            lossVal.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            
            lossValueArray[dataBatch] = lossVal
            
        prof.stop()
        
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

        epochEndTime = time.time()
        logging.info(f"Epoch training elapsed time:{(epochEndTime - epochStartTime)/60.0:3f} mins")
        logging.info("Validating the model")
        # Validation
        with torch.no_grad():
            validationStartTime = time.time()
            logging.info("Setting model to evaluation mode.")
            model = model.eval()
            SE3Dist, eucledianDist, translation, eulerAngle, absEulerAngleErr, absTranslationErr = valitation(model, validatingDataLoader, device)
            validationEndTime = time.time()
            logging.info(f"Validation elapsed time: {(validationEndTime - validationStartTime)/60.0:3f} mins")

        # Scheduler step
        logging.info("Stepping thru scheduler")
        scheduler.step(np.mean(eucledianDist))
        print(f"Mean loss value = {lossVal}")
        logging.info((f"Mean loss value = {lossVal}"))
        print(f"Epoch: {epochs}\t Mean SE3 Dist = {np.mean(SE3Dist):3f} \t Mean Eucledian Distance = {np.mean(eucledianDist):3f}")
        logging.info((f"Epoch: {epochs}\t Mean SE3 Dist = {np.mean(SE3Dist):3f} \t Mean Eucledian Distance = {np.mean(eucledianDist):3f}"))
        print(f"Mean Absolute Errors: Euler angles: x={np.mean(absEulerAngleErr,1)[0]:3f}\u00b0, y={np.mean(absEulerAngleErr,1)[1]:3f}\u00b0, z={np.mean(absEulerAngleErr,1)[2]:3f}\u00b0\
              \t Translation: x={np.mean(absTranslationErr,1)[0]:3f}m, y={np.mean(absTranslationErr,1)[1]:3f}m, z={np.mean(absTranslationErr,1)[2]:3f}m")
        logging.info(f"Mean Absolute Errors: Euler angles: x={np.mean(absEulerAngleErr,1)[0]:3f}\u00b0, y={np.mean(absEulerAngleErr,1)[1]:3f}\u00b0, z={np.mean(absEulerAngleErr,1)[2]:3f}\u00b0\
              \t Translation: x={np.mean(absTranslationErr,1)[0]:3f}m, y={np.mean(absTranslationErr,1)[1]:3f}m, z={np.mean(absTranslationErr,1)[2]:3f}m")
        
        if monitorDist > np.mean(eucledianDist):
            
            monitorDist = np.mean(eucledianDist)
            if not os.path.exists(config.checkpointDir):
                os.makedirs(config.checkpointDir)
            timeStamp = datetime.now()
            checkPointFileName = '-'.join((config.name, str(timeStamp.date()),
                                    str(timeStamp.hour), str(timeStamp.minute),
                                    'Epoch',f"{epochs}","ED",f"{np.mean(eucledianDist):3f}","Inference.pth"))
            
            checkpointPath = os.path.join(config.checkpointDir,checkPointFileName)

            saveCheckPoint(model, optimizer, epochs, lossVal, scheduler, checkpointPath)
            logging.info(f"Milestone Achieved. Saving checkpoint\n \
                 \tEpoch :{epochs}\n \
                 \tLoss: {loss}\n \
                 \tCheckpoint Path: {checkpointPath}\n \
                 \tMonitoring Metric:{monitorDist} " )
        
        if not os.path.exists(config.checkpointDir):
            os.makedirs(config.checkpointDir)
        timeStamp = datetime.now()
        progressiveCheckPointFileName = '-'.join((config.name, str(timeStamp.date()),
                                        str(timeStamp.hour), str(timeStamp.minute),
                                        'Epoch',f"{epochs}","ED",f"{np.mean(eucledianDist):3f}","Progressive.pth"))

        checkPointFilePath = os.path.join(config.checkpointDir,progressiveCheckPointFileName)
        saveCheckPoint(model, optimizer, epochs, lossVal, scheduler, checkPointFilePath)
        if not previousfilePath == None:
            os.remove(previousfilePath)
        previousfilePath = checkPointFilePath



if __name__ == '__main__':
    main()
