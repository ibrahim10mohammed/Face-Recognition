import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.optimizers import SGD
from keras.optimizers import adam
from keras.models import load_model
from keras.preprocessing import image

import os
import numpy as np
from PIL import Image
import tensorflow as tf

from random import shuffle 
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
path = "faces-after-crop\\"

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]   
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        
        id = int(os.path.split(imagePath)[-1].split("-")[1]) ##ibrahim-1-counter.jpg  
        
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

classes = np.unique(train_Y)
nClasses = len(classes)

train_X = train_X.reshape(-1, 48,48, 1)
test_X = test_X.reshape(-1, 48,48, 1)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

train_Y_one_hot = to_categorical(train_Y)

test_Y_one_hot = to_categorical(test_Y)

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
train_label = train_label[:,1:]
valid_label = valid_label[:,1:]
test_Y_one_hot = test_Y_one_hot[:,1:]
batch_size = 64
epochs = 20


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(48,48,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(nClasses, activation='softmax'))
opt = adam(lr=0.001, decay=1e-6)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt,metrics=['accuracy'])

model.summary()

fashion_train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

model.save('cnn_model.h5')
weight = pd.DataFrame(model.layers[0].get_weights()[1])
weight.to_csv("cnnModel.csv")
#B_Output_Hidden = fashion_model.layers[1].get_weights()[1]