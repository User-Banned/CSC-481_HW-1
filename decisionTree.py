import os, math
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC

# Read .pts file and make an array of x and y points
def load_points(filepath):
    return np.loadtxt(filepath, comments=("version:", "n_points:", "{", "}"))

# Euclidian distance
def euclDist(x1,y1,x2,y2):
    return math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))

# Normalization of feature set
def normalize(featureSet):
    normSet = []
    l = featureSet[0]
    s = featureSet[0]
    for person in range(len(featureSet)):
        if l < featureSet[person]:
            l = featureSet[person]
        if s > featureSet[person]:
            s = featureSet[person]
    for p in range(len(featureSet)):
        normSet.append((featureSet[p]-s)/(l-s))
    return normSet

# Retrieve all features from raw data points
# reminder rawData is in form of [face][feature points]
def getFeatures(data,rawData):
    eyeLength=[]
    eyeDistance=[]
    nose=[]
    lipSize=[]
    liplength=[]
    eyebrowLength=[]
    agressive=[]
    for face in range(len(rawData)):
        eyeLength.append(getEyeLength(rawData,face))
        eyeDistance.append(getEyeDistance(rawData,face))
        nose.append(getNose(rawData,face))
        lipSize.append(getLipSize(rawData,face))
        liplength.append(getLipLength(rawData,face))
        eyebrowLength.append(getEyebrowLength(rawData,face))
        agressive.append(getAgressive(rawData,face))
    eyeLength=normalize(eyeLength)
    eyeDistance=normalize(eyeDistance)
    nose=normalize(nose)
    lipSize=normalize(lipSize)
    liplength=normalize(liplength)
    eyebrowLength=normalize(eyebrowLength)
    agressive=normalize(agressive)
    for face in range(len(rawData)):
        data.append([eyeLength[face],eyeDistance[face],
                     nose[face],lipSize[face],liplength[face],
                     eyebrowLength[face],agressive[face]])

# reminder data is in form of [face][feature]
def getTrainTest(train,test,data):
    for face in range(len(data)):
        if face in indexFirst:
            test.append(data[face])
        else:
            train.append(data[face])

### Calculate features: eyeLengthRatio,eyeDistanceRatio, 
#                   noseRatio,lipSizeRatio, lipLengthRatio,
#                   eyebrowLengthRatio, agressiveRatio
def getEyeLength(rD,f):
    x1r,y1r = rD[f][11]
    x2r,y2r = rD[f][12]
    x1l,y1l = rD[f][9]
    x2l,y2l = rD[f][10]
    x3,y3 = rD[f][8]
    x4,y4 = rD[f][13]
    distance1 = euclDist(x1r,y1r,x2r,y2r) + euclDist(x1l,y1l,x2l,y2l)
    distance2 = euclDist(x3,y3,x4,y4)
    return distance1/distance2
def getEyeDistance(rD,f):
    x1,y1 = rD[f][0]
    x2,y2 = rD[f][1]
    x3,y3 = rD[f][8]
    x4,y4 = rD[f][13]
    distance1 = euclDist(x1,y1,x2,y2)
    distance2 = euclDist(x3,y3,x4,y4)
    return distance1/distance2
def getNose(rD,f):
    x1,y1 = rD[f][15]
    x2,y2 = rD[f][16]
    x3,y3 = rD[f][20]
    x4,y4 = rD[f][21]
    distance1 = euclDist(x1,y1,x2,y2)
    distance2 = euclDist(x3,y3,x4,y4)
    return distance1/distance2
def getLipSize(rD,f):
    x1,y1 = rD[f][2]
    x2,y2 = rD[f][3]
    x3,y3 = rD[f][17]
    x4,y4 = rD[f][18]
    distance1 = euclDist(x1,y1,x2,y2)
    distance2 = euclDist(x3,y3,x4,y4)
    return distance1/distance2
def getLipLength(rD,f):
    x1,y1 = rD[f][2]
    x2,y2 = rD[f][3]
    x3,y3 = rD[f][20]
    x4,y4 = rD[f][21]
    distance1 = euclDist(x1,y1,x2,y2)
    distance2 = euclDist(x3,y3,x4,y4)
    return distance1/distance2
def getEyebrowLength(rD,f):
    x1a,y1a = rD[f][4]
    x2a,y2a = rD[f][5]
    x1b,y1b = rD[f][6]
    x2b,y2b = rD[f][7]
    x3,y3 = rD[f][8]
    x4,y4 = rD[f][13]
    distance1a = euclDist(x1a,y1a,x2a,y2a)
    distance1b = euclDist(x1b,y1b,x2b,y2b)
    if distance1a > distance1b:
        distance1 = distance1a
    elif distance1b > distance1a:
        distance1 = distance1b
    distance2 = euclDist(x3,y3,x4,y4)
    return distance1/distance2
def getAgressive(rD,f):
    x1,y1 = rD[f][10]
    x2,y2 = rD[f][19]
    x3,y3 = rD[f][20]
    x4,y4 = rD[f][21]
    distance1 = euclDist(x1,y1,x2,y2)
    distance2 = euclDist(x3,y3,x4,y4)
    return distance1/distance2

# Variables for data, targets, and indexing
allRawData = []         # [face][feature]
allData = []            # all 7 features from all persons normalized
trainData = []          # all but face id 01 from each person
testData  = []          # only from face id 01 from each person
trainTarget = []        # either 'm' or 'w'
testTarget  = []        # either 'm' or 'w'
indexFirst = []         # an index of raw data list for all files having id = 01

### Getting All Paths to .pts Files
# Note all pts files sould be labeled in style like   <m/w>-<001>-<01>
#                                              man/woman - person# - face#
#                                                       - [001-999] - [01-99]
try:    # Try getting data from 'Face Database'
    FD = 'Face Database'
    if FD in os.listdir():  # Find 'Face Database' folder if not found raise exception.
        indexFaceCounter = 0
        for folder in os.listdir(FD+'/'):
            person = os.listdir(FD+'/'+folder+'/')
            for file in person:
                if file[5:8] == '-01':
                    indexFirst.append(indexFaceCounter)
                    allRawData.append(load_points(FD+'/'+folder+'/'+file))
                    testTarget.append(file[0])
                else:
                    allRawData.append(load_points(FD+'/'+folder+'/'+file))
                    trainTarget.append(file[0])
                indexFaceCounter += 1
    else:
        raise("No 'Face Database' Found")
except:
    print("Not a Good Database")

### Get all features: eyeLengthRatio, eyeDistanceRatio, 
#                   noseRatio, lipSizeRatio, lipLengthRatio,
#                   eyebrowLengthRatio, agressiveRatio
getFeatures(allData,allRawData)
getTrainTest(trainData,testData,allData)    # Separate Train and Test Data

# Training and testing phase
dt = DTC(criterion="entropy")
dt.fit(trainData,trainTarget)    # Training Complete
pr = dt.predict(testData)        # Testing Complete

### Results/Analysis
result = []
for res in pr:
    result.append(res)  # Puts precited result into a list so its easier to compare

print('=========== Results ===========')
print('Predicted: ',result)
print('     True: ',testTarget)

# Confusion Matrix
trueMale   = 0
falseMale  = 0
falseWoman = 0
trueWoman  = 0

for i in range(len(result)):
    if   testTarget[i] == result[i] and testTarget[i] == 'm':
        trueMale = trueMale + 1
    elif testTarget[i] != result[i] and testTarget[i] == 'w':
        falseMale = falseMale + 1
    elif testTarget[i] != result[i] and testTarget[i] == 'm':
        falseWoman = falseWoman + 1
    elif testTarget[i] == result[i] and testTarget[i] == 'w':
        trueWoman = trueWoman + 1

print('\n ','m','w')
print('m',trueMale,falseWoman)
print('w',falseMale,trueWoman,'\n')

# Accuracy,Precision,Recall
print('Accuracy: ',(trueMale+trueWoman)/(trueMale+falseMale+falseWoman+trueWoman),'\n')
print('Precision of m: ',(trueMale)/(trueMale+falseMale))
print('Precision of w: ',(trueWoman)/(falseWoman+trueWoman))
print('Precision Average: ',((trueMale)/(trueMale+falseMale)+(trueWoman)/(falseWoman+trueWoman))/2,'\n')
print('Recall of m: ',(trueMale)/(trueMale+falseWoman))
print('Recall of w: ',(trueWoman)/(falseMale+trueWoman))
print('Recall Average: ',((trueMale)/(trueMale+falseWoman)+(trueWoman)/(falseMale+trueWoman))/2)
