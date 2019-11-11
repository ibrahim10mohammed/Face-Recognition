from keras.preprocessing import image
import numpy as np
from PIL import Image
from numpy import zeros, newaxis
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('neural_network.h5')

import webbrowser



#loaded_model = keras.models.load_model('neural_network.h5')
Labels_Names = {"1":webbrowser.open('your facebook url for example') ,"2":webbrowser.open(''your facebook url for example')}
def predict_image_Neural_Network(image_path):
    
    PIL_img = Image.open(image_path).convert('L')
    img_numpy = np.array(PIL_img,'uint8')
    
    img_numpy = img_numpy [newaxis ,:, :]
    
    images = np.vstack([img_numpy])
    classes = model.predict_classes(images, batch_size=64)
    return (classes)
print("                     Predicate Neural Network ")
image_path = input("Enter the path of Image : ")
class_predicted = predict_image_Neural_Network(image_path)
print(Labels_Names[str(class_predicted[0])])


print("                     Predicate convolutional Neural Network ")
#AhmedSamir-4-36.jpg
model = load_model('cnn_model.h5')
def predict_image_CNN(image_path):    
    PIL_img = Image.open(image_path).convert('L')
    img_numpy = np.array(PIL_img,'uint8')    
    img_numpy=img_numpy.reshape(-1, 48,48, 1)
    images = np.vstack([img_numpy])
    classes = model.predict_classes(images, batch_size=64)
    return (classes+1)

class_predicted = predict_image_CNN(image_path)
print(Labels_Names[str(class_predicted[0])])

