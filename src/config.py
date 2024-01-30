import os
datasetFile = "/mnt/data/kitti/calibration_20_0.1/calibData.json"
checkpointDir = os.path.join(os.getcwd(), 'checkpoints')
logDir = os.path.join(os.getcwd(),'logs')
DDPT = False # Distributed Data Parallel Training
DPT = False # Data Parallel Training
training = dict(
    batchSize = 30,
    epoch = 25,
    learningRate = 0.005,
    beta0 = 0.9,
    beta1 = 0.999,
    eps = 1e-08,
    decayRate = 1e-4
)

