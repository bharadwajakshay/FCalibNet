import os
name = "FCalib_w_UNet3D"
mode = "Train" # Train / # Evaluate
datasetFile = "/home/akshay/kitti_Dataset_40_1/calibData.json"
checkpointDir = os.path.join(os.getcwd(), 'checkpoints')
logDir = os.path.join(os.getcwd(),'logs')
DDPT = False # Distributed Data Parallel Training
DPT = False # Data Parallel Training
modelParallel = True # Need this for Unet3D
optimizer = "SGD" # 'Adam'/"SGD"

# Scheduler  deteails 
# ROP = Reduce on Platue
scheduler = dict(
    name="ROP", # 'ROP'S
    mode = 'min', # 'max'/'max'
    patience = 4, # integer values
    step = [24,25]
) 
training = dict(
    batchSize = 25,
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
    "CHORDAL" = Chordal distance between the predicted and the target transforms
    "EUCTR" = Euclidean distance between the translation of the predicated transform and the ground truth
'''

loss = ["WEUC","CHORDAL","EUCTR"]

# Loss weight needs to be the same size as loss 
lossWeight = [1.0, 0.7, 0.7]

