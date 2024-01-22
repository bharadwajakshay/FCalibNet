import numpy as np
from PIL import Image
import json
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

_debug = False



class dataLoader(Dataset):
    def __init__ (self, filename,  mode='train'):
        self.datafile = filename
        self.colorImageHeight = 224
        with open(self.datafile,'r') as jsonFile:
            self.data = json.load(jsonFile)

        dataList = []

        for keys in list(self.data.keys()):
            for scenes in  list(self.data[keys].keys()):
                dataList.append(self.data[keys][scenes])

        trainIdx = int(len(dataList)*0.75)
        testIdx = int(len(dataList)*0.9)

        if mode =='train':
            self.data = dataList[:trainIdx]
           
        if mode =='test':
            self.data = dataList[trainIdx:testIdx]
       
        if mode =='evaluate':
            self.data = dataList[testIdx:]

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, key):
        return(self.getItem(key))
    
    def readimgfromfilePIL(self,pathToFile,resizeRows=None):
        image = Image.open(pathToFile)
        image = image.convert('RGB')

        if resizeRows != None:
            resizeRatio = resizeRows/image.height
            height = resizeRows
            width = image.width * resizeRatio
            image = image.resize((int(np.round(width)),height),Image.BICUBIC)

        return(image.width,image.height,image)
    
    def readProjectionData(self,pathToFile,channels):
        projectionData = np.fromfile(pathToFile,dtype=np.float32).reshape(64,-1,channels)
        return projectionData


    
    def getItem(self,key):
        
        sample = self.data[key]

        srcImageFilename = sample['image filename']
        srcLiDARDepthFilename = sample['depth projection filename']
        srcLiDARIntensityFilename = sample['intensity projection filename']
        srcLiDARNormalsFilename = sample['normals projection filename']
        pointsIndecies = sample['point indices of projection filename']
        transfrom =  np.asarray(sample['total transformation'])

        __, __, srcClrImg = self.readimgfromfilePIL(srcImageFilename,self.colorImageHeight)
        colorImage = np.array(srcClrImg, dtype=np.float32)

        lidarDepthImage = self.readProjectionData(srcLiDARDepthFilename, 1)
        lidarIntensityImage = self.readProjectionData(pointsIndecies, 1)
        lidarNormalImage = self.readProjectionData(srcLiDARNormalsFilename, 4)[:,:,:3]

        netLiDARImage = np.concatenate((lidarNormalImage,lidarDepthImage,lidarIntensityImage),axis=2)

        if _debug == True:
            ## Verification of range image
            # GEt color map
            colorMap = plt.get_cmap('rainbow')
            lidarDepthImage = lidarDepthImage*(255/lidarDepthImage.max())
            lidarDepthImage[lidarDepthImage < 0] = 0
            colorMappedImage = colorMap(lidarDepthImage.squeeze(axis=2))
            image = Image.fromarray((lidarDepthImage[:,:,0]).astype(np.uint8)).save('depthIMage.png')

            # intensity image
            lidarIntensityImage = np.nan_to_num(lidarIntensityImage)
            colorMap = plt.get_cmap('rainbow')
            colorMappedImage = colorMap(lidarIntensityImage.squeeze(axis=2))
            image = Image.fromarray((lidarIntensityImage[:,:,0]*255).astype(np.uint8)).save('intensityImage.png')

            srcClrImg.save('colorImage.png')


        return([colorImage, netLiDARImage, transfrom])
