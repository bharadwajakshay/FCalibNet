import os
import json
from dataLoader import dataLoader
import logging
import config
import torch
from tqdm import tqdm
from common import tensorTools
import numpy as np 



def checkForTransformation(pc,pcGT,rt):
    
    # Get inverse transform of the transformation
    invTransformMat = tensorTools.calculateInvRTTensorWhole(rt)
    transPC = tensorTools.applyTransformationOnTensor(pc[:,:,:,:3], invTransformMat)
    transPC = transPC.squeeze(0).detach().numpy()
    pcGT = pcGT[:,:,:,:3].squeeze(0).detach().numpy()

    # Caluclate euclidean distance between GT and transformed pointcloud.
    # The eucledean distance has to be zeo or close to 
    dist = np.linalg.norm(transPC - pcGT,ord=2,axis=2)

    return dist


def main():
    testingData = dataLoader(config.datasetFile,"whole")
    batchsize = 1
    testDataLoader = torch.utils.data.DataLoader(testingData,
                                                     batch_size=batchsize,
                                                     shuffle=True,
                                                     drop_last=True)
    eucledianDist = []
    for dataBatch, data in tqdm(enumerate(testDataLoader,0),total=len(testDataLoader)):
        __,pc,pcGT,rt,__ = data

        eucledianDist.append(checkForTransformation(pc, pcGT, rt).mean())

    print(f"The collective mean of euclidean distance is:{sum(eucledianDist)/(len(eucledianDist))}")
    if all(i >=0.00001 for i in eucledianDist):
        print("There seems to be aproblemin the data")
    else:
        print("Generated data with transformations seems ok")







if __name__ == '__main__':
    main()