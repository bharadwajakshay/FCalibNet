import numpy as np
# Moving from PIL to opnvCV as PIL is much slower
# from PIL import Image
import cv2
import json
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time
import logging

_debug = False



class dataLoader(Dataset):
    def __init__ (self, filename,  mode='train'):
        logging.info(f"initializing data loader.")

        self.datafile = filename
        self.colorImageHeight = 224
        self.colorImageWidth = 740
        with open(self.datafile,'r') as jsonFile:
            self.data = json.load(jsonFile)


        trainIdx = int(len(self.data)*0.8)
        #trainIdx = 100

        testIdx = int(len(self.data)*0.9)
        #testIdx = 200

        if mode =='train':
            self.data = self.data[:trainIdx]
           
        if mode =='test':
            self.data = self.data[trainIdx:testIdx]
       
        if mode =='evaluate':
            self.data = self.data[testIdx:]

        logging.info(f"Dataloader details.\n \
                     \t Filename:{filename}\n \
                     \t Image Dimensions: {self.colorImageHeight}x{self.colorImageWidth}\n \
                     \t Mode: {mode}\n \
                     \t Data Length: {len(self.data)}")

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, key):
        return(self.getItem(key))
    
    def readimgfromfilePIL(self,pathToFile,resizeRows=None, resizeCols=None):
        # Moving from PIL to openCV
        
        image = cv2.imread(pathToFile,cv2.IMREAD_COLOR )
    
        if resizeRows != None and resizeCols != None:
            image = cv2.resize(image,(resizeCols,resizeRows),cv2.INTER_LINEAR)

        elif resizeRows != None and resizeCols == None:
            resizeRatio = resizeRows/image.shape[0]
            height = resizeRows
            width = image.shape[1] * resizeRatio
            image = cv2.resize(image,(int(np.round(width)),height),cv2.INTER_LINEAR)
        elif resizeRows == None and resizeCols != None:
            resizeRatio = resizeCols/image.shape[1]
            width = resizeCols
            height = image.shape[0] * resizeRatio
            image = cv2.resize(image,(width,int(np.round(height))),cv2.INTER_LINEAR)

        return(image.shape[1],image.shape[0],image)
    
    def readProjectionData(self,pathToFile):
        projectionData = np.fromfile(pathToFile,dtype=np.float64)
        
        return projectionData.reshape([64,900,-1])
    
    def readProjectionDataMemmap(self,pathToFile):        
        return np.memmap(pathToFile,dtype=np.float64, mode='r', shape=(64,900,5))
    
    def readTransform(self,strTransform, mode='TR'):
        transform = strTransform.replace('[','').replace(']','').split('\n')
        if mode == 'TR':
            matsize = [4,4]
        else:
            matsize = [3,4]
        mat = np.zeros(matsize)
        count = 0
        for each in transform:
            data = each.split(' ')
            array = []
            for eachelement in range(0,len(data)):
                if data[eachelement] != '':
                    array.append(float(data[eachelement]))

            mat[count,:] = np.asarray(array)
            count+=1

        return mat
        
    
    def getItem(self,key):
        
        #getItemStartTime = time.time()
        
        sample = self.data[key]

        trainImgFilename = sample['imageFP']
        trainProjectedPC = sample['deCalibDataFP']
        gndTruthPC = sample['groundTruthDataFP']
        transfromRT = np.asarray(sample['transformationMat'])
        projectMat = np.asarray(sample['projectionMat'])
                
        __, __, srcClrImg = self.readimgfromfilePIL(trainImgFilename, self.colorImageHeight, self.colorImageWidth)
        #__, __, srcClrImg = self.readimgfromfilePIL(trainImgFilename)
        colorImage = np.array(srcClrImg, dtype=np.float32)

        trainPointCloud = self.readProjectionDataMemmap(trainProjectedPC)

        goundTruthPointCloud = self.readProjectionDataMemmap(gndTruthPC)


        if _debug == True:
            ## Verification of range image
            # display range image from transformed pointcloud and GT point cloud
            rangeTransformed = np.linalg.norm(trainPointCloud[:,:,:3], 2, axis=2)
            rangeTransformed = rangeTransformed * (255/rangeTransformed.max())
            Image.fromarray(rangeTransformed.astype(np.uint8)).save('distortedPCRange.png')
            
            rangeGT = np.linalg.norm(goundTruthPointCloud[:,:,:3], 2, axis=2)
            rangeGT[rangeGT<0] = 0
            rangeGT = rangeGT * (255/rangeTransformed.max())
            Image.fromarray(rangeGT.astype(np.uint8)).save('groundTruthPCRange.png')
            
            intensityTransformed = trainPointCloud[:,:,-1]
            intensityTransformed[intensityTransformed < 0 ] = 0
            intensityTransformed = intensityTransformed *255
            Image.fromarray(intensityTransformed.astype(np.uint8)).save('distortedPCIntensity.png')
            
            
            #colormap = plt.get_cmap('rainbow')
            
            
        #getItemEndTime = time.time()
        #print(f"total timeElapsed={getItemEndTime - getItemStartTime}, \
        #        time to fetch color image={getcolorEndTime - getcolorStartTime}, \
        #        time to fetch 1st PC = {getpointsEndTime - getcolorEndTime}, \
        #        time to fetch 1st PC MemMap = {getpointsmemMapEndTime - getpointsEndTime} \
        #        time to fetch 2nd PC = {getgtpointsEndTime - getpointsmemMapEndTime}\
        #        time to fetch 2nd PC MemMap= {getgtpointsmemMapEndTime - getgtpointsEndTime}")


        return([colorImage, trainPointCloud, goundTruthPointCloud, transfromRT, projectMat])
