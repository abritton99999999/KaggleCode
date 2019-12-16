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

os.chdir('/depot/wwtung/data/Brittoa/Kaggle/understanding_cloud_organization')
Optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# load json and create model Sugar
json_file = open('Sugar3000Model.json', 'r')
Sugar_model_json = json_file.read()
json_file.close()
Sugar_model = model_from_json(Sugar_model_json)
Sugar_model.load_weights("Sugar3000Model10.h5")
print("Loaded model from disk: Sugar")
 
# evaluate loaded model on test data
Sugar_model.compile(loss='mse', optimizer= Optimizer)


Images = sorted(os.listdir("test_images"))
Submission = pd.read_csv("Submission.csv")

for Image in range(0,int(round(len(Images)))):
    print(Image)
    dataInput =cv2.imread('test_images/' + Images[Image])
    dataInput = np.array(cv2.resize(dataInput, (512, 256)))
    dataInput = dataInput / 255
    dataInput = dataInput.reshape(-1, 256,512,3)
    SugarPredict = Sugar_model.predict(dataInput)
    SugarPredict = Compressor(SugarPredict[0])
    Submission.loc[(4*Image)+3, "EncodedPixels"] =SugarPredict
    if Image%100 == 0 or Image == len(Images)-1:
        Submission.to_csv("SubmissionTwo.csv")
        print("Submission Saved:", Image, "of", len(Images))
