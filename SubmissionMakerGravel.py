import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import glob
import pandas as pd
from keras import backend as K
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from tensorflow.keras import layers
import scipy
print("test")

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
print("test")

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
print("test")

def Compressor(Input):
    Output = [];
    for pixel in range(0, Input.shape[1]):
        print(pixel)
        for value in range(0, Input.shape[0]):
            print(value)
            if (Input[value, pixel] >= .2):
                Output.append(pixel*Input.shape[0]+value);
            #print(pixel*Input.shape[2]+value)
            #print(Input[pixel])
    print(len(Output))
    print(Output)
    x = 0;
    OutputPix = ""
    runtime = 1;
    for pixel in range(0, len(Output)):
        if x == 0:
            OutputPix = OutputPix + " " + str(Output[pixel]) + " "
            x = x + 1;
        if (Output[pixel] - Output[pixel-1]) <= 5 :
            runtime = runtime + 1;
        else:
            OutputPix = OutputPix + str(runtime)
            #print(Output[pixel-1]-Output[pixel])
            x=0
            runtime  = 1;
    if len(OutputPix) < 10:
        OutputPix = "1 1"
    if pixel == len(Output) -1 and len(OutputPix) >= 10:
        OutputPix = OutputPix + str(runtime)
    #print(Input)
    return(str(OutputPix))
print("test")

os.chdir('/depot/wwtung/data/Brittoa/Kaggle/understanding_cloud_organization')
Optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# load json and create model Sugar
#json_file = open('Sugarmodel2.json', 'r')
#print("Loaded model from disk: Sugar")

#Sugar_model_json = json_file.read()
#print("Loaded model from disk: Sugar")

#json_file.close()
#print("Loaded model from disk: Sugar")

#Sugar_model = model_from_json(Sugar_model_json)
#print("Loaded model from disk: Sugar")

# load weights into new model
#Sugar_model.load_weights("Sugarmodel2.h5")
#print("Loaded model from disk: Sugar")
 
# evaluate loaded model on test data
#Sugar_model.compile(loss=dice_coef_loss, metrics=[dice_coef], optimizer=keras.optimizers.Adam())

# load json and create model Flower
#json_file = open('Flowermodel3.json', 'r')
#Flower_model_json = json_file.read()
#json_file.close()
#Flower_model = model_from_json(Flower_model_json)
# load weights into new model
#Flower_model.load_weights("Flowermodel3.h5")
#print("Loaded model from disk: Flower")
 
# evaluate loaded model on test data
#Flower_model.compile(loss=dice_coef_loss, metrics=[dice_coef], optimizer=keras.optimizers.Adam())

# load json and create model Fish
#json_file = open('Fishmodel3.json', 'r')
#Fish_model_json = json_file.read()
#json_file.close()
#Fish_model = model_from_json(Fish_model_json)
# load weights into new model
#Fish_model.load_weights("Fishmodel3.h5")
#print("Loaded model from disk: Fish")
 
# evaluate loaded model on test data Fish
#Fish_model.compile(loss=dice_coef_loss, metrics=[dice_coef], optimizer=keras.optimizers.Adam())

# load json and create model Gravel
json_file = open('Gravel250Model.json', 'r')
Gravel_model_json = json_file.read()
json_file.close()
Gravel_model = model_from_json(Gravel_model_json)
# load weights into new model
Gravel_model.load_weights("Gravel250Model10.h5")
print("Loaded model from disk: Gravel")
 
# evaluate loaded model on test data
Gravel_model.compile(loss='mse', optimizer= Optimizer)



Images = sorted(os.listdir("test_images"))
Submission = pd.read_csv("Submission.csv")

for Image in range(0,int(round(len(Images)))):
    print(Image)
    #Name = Submission.loc[4*Image, "Image_Label"]
    #print(Name)
    #print(Name[1])
    #print(Images[Image])
    dataInput =cv2.imread('test_images/' + Images[Image])
    dataInput = np.array(cv2.resize(dataInput, (512, 256)))
    dataInput = dataInput / 255
    dataInput = dataInput.reshape(-1, 256,512,3)
    #print(dataInput.mean())
    #SugarPredict = Sugar_model.predict(dataInput)
    #FlowerPredict = Flower_model.predict(dataInput)
    GravelPredict = Gravel_model.predict(dataInput)
    #FishPredict = Fish_model.predict(dataInput)
    #print(np.sum(SugarPredict))
    #print(np.sum(FlowerPredict))
    #print(np.sum(GravelPredict))
    #print(argmax(FishPredict))
    #print(SugarPredict[0])
    #SugarPredict = Compressor(SugarPredict[0])
    #FlowerPredict = Compressor(FlowerPredict[0])
    GravelPredict = Compressor(GravelPredict[0])
    #FishPredict = Compressor(FishPredict[0])
    #SugarPredictT = str(SugarPredict)
    #print(len(FlowerPredict))
    #Submission.loc[(4*Image)+3, "EncodedPixels"] =SugarPredict
    #Submission.loc[4*Image, "EncodedPixels"] =FishPredict
    #Submission.loc[(4*Image)+1, "EncodedPixels"] =FlowerPredict
    Submission.loc[(4*Image)+2, "EncodedPixels"] =GravelPredict
    if Image%100 == 0 or Image == len(Images)-1:
        Submission.to_csv("Submission.csv")
        print("Submission Saved:", Image, "of", len(Images))
    #print(Submission.loc[4*Image+1, "EncodedPixels"])
    #ur mom;