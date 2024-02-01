import os
datasetFile = "/mnt/data/kitti/calibration_20_0.1/calibData.json"
checkpointDir = os.path.join(os.getcwd(), 'checkpoints')
logDir = os.path.join(os.getcwd(),'logs')
DDPT = False # Distributed Data Parallel Training
DPT = False # Data Parallel Training
training = dict(
    batchSize = 20,
    epoch = 25,
    learningRate = 0.005,
    beta0 = 0.9,
    beta1 = 0.999,
    eps = 1e-08,
    decayRate = 1e-4,
)
'''
    # Possible value are 
    "EUC" = Euclidean distance between points
    "WEUC" = Weighted Euclidean distance between points
    "CHAMP" = Champher distance between points
    "EMD" = Earthmovers distance between points
    "CHORDAL" = Chordal distance between the predicted and the target transforms
    "EUCTR" = Euclidean distance between the translation of the predicated transform and the ground truth

 
'''

loss = ["CHAMP","CHORDAL","EUCTR"]

# Loss weight needs to be the same size as loss 
lossWeight = [1.0, 0.7, 0.7]

