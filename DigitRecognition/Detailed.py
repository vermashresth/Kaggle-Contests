import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow  as tf

LEARNING_RATE=1e-4

TRAINING_ITERATIONS=2500

DROPOUT=0.5
BATCH_SIZE=50

VALIDATION_SIZE=2000

IMAGE_TO_DISPLAY=10
# The competition datafiles are in the directory ../input
# Read competition data files:
data = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
print(data.head())

print("data {0[0]}  {0[1]}".format(data.shape))

images=data.iloc[:1:].values
images=images.astype(np.float)

images=np.multiply(images,1.0/255.0)

print('images({0[0]},{0[1]})'.format(images.shape))

image_size=images.shape[1]

image_width=image_height=np.ceil(np.sqrt(image_size)).astype(np.uint8)

def display(img):
    one_image=img.reshape(image_width,image_height)
    
    plt.axis('off')
    plt.imshow(one_image,cmap=cm.binary)
    plt.savefig("output.png")

display(images[0])

labels_flat=data[[0]].values.ravel()
labels_count=np.unique(labels_flat).shape[0]

def dense_to_one_hot(labels_dense,num_classes):
    num_labels=labels_dense.shape[0]
    index_offset=np.arrange(num_labels)*num_classes
    labels_one_hot=np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()]=1
    return labels_one_hot
    
labels=dense_to_one_hot(labels_flat,labels_count)
labels=labels.astype(np.uint8)

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]























