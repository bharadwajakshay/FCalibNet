import os
datasetFile = "/home/akshay/kitti_Dataset_40_1/calibData.json"
checkpointDir = os.path.join(os.getcwd(), 'checkpoints')
logDir = os.path.join(os.getcwd(),'logs')
DDPT = False # Distributed Data Parallel Training
DPT = False # Data Parallel Training
modelParallel = True # Need this for Unet3D
training = dict(
    batchSize = 25,
    epoch = 50,
    learningRate = 0.005,
    beta0 = 0.9,
    beta1 = 0.999,
    eps = 1e-08,
    decayRate = 1e-4,
    loadFromCheckpoint = False
)
chkPointFileName = '2024_2_3_21:18_ED_2.0538178968429563.pth'
'''
    # Possible value are 
    "EUC" = Euclidean distance between points
    "WEUC" = Weighted Euclidean distance between points
    "CHAMP" = Champher distance between points
    "EMD" = Earthmovers distance between points
    "CHORDAL" = Chordal distance between the predicted and the target transforms
    "EUCTR" = Euclidean distance between the translation of the predicated transform and the ground truth
'''

loss = ["WEUC","CHORDAL","EUCTR"]

# Loss weight needs to be the same size as loss 
lossWeight = [1.0, 0.7, 0.7]

