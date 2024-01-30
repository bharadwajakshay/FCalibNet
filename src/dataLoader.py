import numpy as np
from PIL import Image
import json
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time

_debug = False



class dataLoader(Dataset):
    def __init__ (self, filename,  mode='train'):
        self.datafile = filename
        self.colorImageHeight = 224
        self.colorImageWidth = 740
        with open(self.datafile,'r') as jsonFile:
            self.data = json.load(jsonFile)

        

        trainIdx = int(len(self.data)*0.8)
        #trainIdx = 100

        testIdx = int(len(self.data)*0.9)


        if mode =='train':
            self.data = self.data[:trainIdx]
           
        if mode =='test':
            self.data = self.data[trainIdx:testIdx]
       
        if mode =='evaluate':
            self.data = self.data[testIdx:]

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, key):
        return(self.getItem(key))
    
    def readimgfromfilePIL(self,pathToFile,resizeRows=None, resizeCols=None):
        image = Image.open(pathToFile)
        image = image.convert('RGB')

        if resizeRows != None and resizeCols != None:
            image = image.resize((resizeCols,resizeRows),Image.BICUBIC)

        elif resizeRows != None and resizeCols == None:
            resizeRatio = resizeRows/image.height
            height = resizeRows
            width = image.width * resizeRatio
            image = image.resize((int(np.round(width)),height),Image.BICUBIC)
        else:
            resizeRatio = resizeCols/image.width
            width = resizeCols
            height = image.height * resizeRatio
            image = image.resize((width,int(np.round(height))),Image.BICUBIC)

        return(image.width,image.height,image)
    
    def readProjectionData(self,pathToFile):
        projectionData = np.fromfile(pathToFile,dtype=np.float64)
        
        return projectionData.reshape([64,900,-1])
    
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
        
        getItemStartTime = time.time()
        
        sample = self.data[key]

        trainImgFilename = sample['imageFP']
        trainProjectedPC = sample['deCalibDataFP']
        gndTruthPC = sample['groundTruthDataFP']
        transfromRT = self.readTransform(sample['transformationMat'])
        projectMat = self.readTransform(sample['projectionMat'],'PR')
        
        getcolorStartTime = time.time()
                
        __, __, srcClrImg = self.readimgfromfilePIL(trainImgFilename, self.colorImageHeight, self.colorImageWidth)
        colorImage = np.array(srcClrImg, dtype=np.float32)
        getcolorEndTime = time.time()

        trainPointCloud = self.readProjectionData(trainProjectedPC)
        getpointsEndTime = time.time()
        
        goundTruthPointCloud = self.readProjectionData(gndTruthPC)
        getgtpointsEndTime = time.time()
        

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
            
            
            colormap = plt.get_cmap('rainbow')
            
            
        getItemEndTime = time.time()
        #print(f"total timeElapsed={getItemEndTime - getItemStartTime}, \
        #        time to fetch color image={getcolorEndTime - getcolorStartTime}, \
        #        time to fetch 1st PC = {getpointsEndTime - getcolorEndTime}, \
        #        time to fetch 2nd PC = {getgtpointsEndTime - getpointsEndTime}")

        return([colorImage, trainPointCloud, goundTruthPointCloud, transfromRT, projectMat])
