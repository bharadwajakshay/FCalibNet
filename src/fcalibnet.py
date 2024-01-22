import torch

import numpy as np
from tqdm import tqdm
import os
import datetime
from model.model import fCalibNet
from dataLoader import dataLoader
from datetime import datetime

import config

def main():

    if os.path.exists(config.checkpointDir):
        os.makedirs(config.checkpointDir)

    if os.path.exists(config.logDir):
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


    trainingData = dataLoader(config.datasetFile)
    trainingDataLoader = torch.utils.data.DataLoader(trainingData,batch_size=config.training['batchSize'], shuffle=True, num_workers=0,drop_last=True)

    for epochs in range(0,config.training['epoch']):
        # put model in training mode
        model = model.train()


        #  get your data
        for dataBatch, data in tqdm(enumerate(trainingDataLoader,0),total=len(trainingDataLoader)):
            colorImage, lidarImage, gtTransfrom = data
            colorImage = torch.transpose(colorImage.to(device),1,3)
            lidarImage = torch.transpose(lidarImage.to(device),1,3)
            gtTransfrom = gtTransfrom.to(device)
            model(colorImage, lidarImage)




if __name__ == '__main__':
    main()
