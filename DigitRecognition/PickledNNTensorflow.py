import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from os.path import isfile, isdir


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

trainLabelCounts=train["label"].value_count()
print(trainLabelCounts)

def getImage(data, *args):
    '''
    Get the image by specified number (Randomly)
    parameters:
        data: dataframe
        number: int, the label of the number to show
    output: 1-D numpy array
    '''
    if args:
        number = args[0]
        specifiedData = data[data['label'] == number].values
    else:
        specifiedData = data.values
    randomNumber = np.random.choice(len(specifiedData)-1, 1)
    return specifiedData[randomNumber,:]

def plotNumber(imageData, imageSize):
    '''
    parameters:
        data: label & 1-D array of pixels
    '''
    # show the image of the data
    if imageData.shape[1] == np.prod(imageSize):
        image = imageData[0,:].reshape(imageSize)
    elif imageData.shape[1] > np.prod(imageSize):
        label = imageData[0,0]
        image = imageData[0,1:].reshape(imageSize)
        plt.title('number: {}'.format(label))
    plt.imshow(image)
    plt.savefig("check.png")
    
    
    
def plotNumber(imageData, imageSize):
    '''
    parameters:
        data: label & 1-D array of pixels
    '''
    # show the image of the data
    if imageData.shape[1] == np.prod(imageSize):
        image = imageData[0,:].reshape(imageSize)
    elif imageData.shape[1] > np.prod(imageSize):
        label = imageData[0,0]
        image = imageData[0,1:].reshape(imageSize)
        plt.title('number: {}'.format(label))
    plt.imshow(image)
    plt.savefig("check.png")
    
trainData=train.values[:,1:]
trainLabel=train.values[:,0]
testData=test.values

def preprocessing(data):
    minV=0
    maxV=255
    data=(data-minV)/(maxV-minV)
    return data


def one_hot_encoding(data,numberOfClass):
    from sklearn import preprocessing
    lb=preprocessing.LabelBinarizer()
    lb.fit(range(numberOfClass))
    return lb.transform(data)


processedTrainData=preprocessing(trainData)
processedTestData=preprocessing(testData)
one_hot_trainlabel=one_hot_encoding(trainLabel,10)


fileName='mnist.p'
if not isfile(filename):
    pickle.dump((processedTrainData, trainLabel, one_hot_trainLabel, processedTestData),open(fileName,'wb'))
    
trainData, trainLabel, one_hot_trainLabel, testData = pickle.load(open(fileName, mode = 'rb'))



    

