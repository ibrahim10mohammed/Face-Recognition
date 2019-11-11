import tensorflow as tf

from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.optimizers import SGD
from keras.optimizers import adam
from keras.models import load_model
import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image


from random import shuffle 
from keras.utils import to_categorical
import matplotlib.pyplot as plt

path = "faces-after-crop\\"

recogniser = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
path = "faces-after-crop\\"

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split("-")[1])       
        faceSamples.append(img_numpy)
        ids.append(id)
    
    combined = list(zip(faceSamples, ids))
    shuffle(combined)
    faceSamples[:], ids[:] = zip(*combined)
    return np.array(faceSamples), np.array(ids)
    
data,label = getImagesAndLabels(path)

train_X=data[:100]
train_Y=label[:100]
test_X=data[100:]
test_Y=label[100:]

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

train_Y_one_hot = to_categorical(train_Y)

test_Y_one_hot = to_categorical(test_Y)

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=train_X[0].shape),
        tf.keras.layers.Dense(512,activation=tf.nn.relu),
        tf.keras.layers.Dense(256,activation=tf.nn.relu),
        tf.keras.layers.Dense(256,activation=tf.nn.relu),
        tf.keras.layers.Dense(4,activation=tf.nn.softmax)
        ])
model.compile(optimizer ='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_X,train_Y,epochs=25, batch_size =80)
test_eval=model.evaluate(test_X, test_Y)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])








model.save("neural_network.h5")