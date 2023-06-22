import numpy as np
from PIL import Image
import json
from torch.utils.data import Dataset

class dataLoader(Dataset):
    def __init__ (self, filename,  mode='train'):
        self.datafile = filename
        with open(self.datafile,'r') as jsonFile:
            self.data = json.load(jsonFile)

        dataList = []

        for keys in list(self.data.keys()):
            for scenes in  list(self.data[keys].keys()):
                self.dataList.append(self.data[keys][scenes])

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
    
    def readimgfromfilePIL(self,pathToFile):
        image = Image.open(pathToFile)
        image = image.convert('RGB')
        return(image.width,image.height,image)
    
    def readProjectionData(self,pathToFile):
        projectionData = np.fromfile(pathToFile)
        breakpoint


    
    def getItem(self,key):
        
        sample = self.data[key]

        srcImageFilename = sample['image filename']
        srcLiDARDepthFilename = sample['depth projection filename']
        srcLiDARIntensityFilename = sample['intensity projection filename']
        srcLiDARNormalsFilename = sample['normals projection filename']
        pointsIndecies = sample['point indices of projection filename']
        transfrom =  np.asarray(sample['transfromation'])

        __, __, srcClrImg = self.readimgfromfilePIL(srcImageFilename)
        colorImage = np.array(srcClrImg)

    













