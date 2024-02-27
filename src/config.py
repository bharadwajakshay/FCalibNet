import os
name = "FCalib_w_RESNET_n_Crossattention(profile)"
mode = "Train" # Train / # Evaluate
datasetFile = "/home/akshay/kitti_Dataset_40_1/calibData.json"
checkpointDir = os.path.join(os.getcwd(), 'checkpoints')
logDir = os.path.join(os.getcwd(),'logs')
DDPT = False # Distributed Data Parallel Training
DPT = False # Data Parallel Training
modelParallel = False # Need this for Unet3D
optimizer = "Adam" # 'Adam'/"SGD"

# Scheduler  deteails 
# ROP = Reduce on Platue
scheduler = dict(
    name='MLR', # 'ROP'/'MLR'
    mode = 'min', # 'max'/'max'
    patience = 4, # integer values
    step = [24,25]
) 
training = dict(
    batchSize = 30,
    epoch = 30,
    learningRate = 0.001,
    beta0 = 0.9,
    beta1 = 0.999,
    eps = 1e-08,
    decayRate = 1e-4,
    loadFromCheckpoint = False,
    momentum = 0.9
)
chkPointFileName = '2024_2_3_21:18_ED_2.0538178968429563.pth'
'''
    # Possible value are 
    "EUC" = Euclidean distance between points
    "WEUC" = Weighted Euclidean distance between points
    "CHAMP" = Champher distance between points
    "EMD" = Earthmovers distance between points
    "CHORDAL" = Chordal distance between the predicted and the target SO(3) rotations
    "EUCTR" = Euclidean distance between the translation of the predicated transform and the ground truth
    "MANHATTAN" = Manhattan distance or the L1 norm between the points
    "GEODESIC" = Geodesic distance between the predicted and the target SO(3) rotations
'''

loss = ["GEODESIC","EUCTR"]

# Loss weight needs to be the same size as loss 
lossWeight = [2,1]

