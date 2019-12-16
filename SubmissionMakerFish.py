import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.models import model_from_json
import cv2
from scipy import gmean
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers import BatchNormalization,Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import keras
import glob
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

print("test")

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

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
#Optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# load json and create model Fish
json_file = open('Fish250Model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("Fish250Model10.h5")
print("Loaded model from disk: Fish")
 
# evaluate loaded model on test data Fish
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])


Images = sorted(os.listdir("test_images"))
Submission = pd.read_csv("Submission.csv")

for Image in range(0,int(round(len(Images)))):
    print(Image)
    img = load_img("test_images/" + Images[Image])
    X = img_to_array(img)
    dataInput = resize(X, (256,512))
    X= dataInput/255
    X = X.reshape(-1, 256,512,3)
    Predict = model.predict(X)
    print(gmean(Predict[0]))
    Predict = Compressor(Predict[0])
    Submission.loc[4*Image, "EncodedPixels"] =Predict
    if Image%100 == 0 or Image == len(Images)-1:
        Submission.to_csv("Submission.csv")
        print("Submission Saved:", Image, "of", len(Images))