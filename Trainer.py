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
from scipy import stats

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
import skimage.segmentation as seg

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

import glob
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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
def unet(pretrained_weights = None,input_size = (256,512,3)):
    from keras.applications import VGG16

    #conv_base = VGG16(weights='imagenet',
                  #include_top=False,
                  #input_shape=(1024, 2048, 3))
   
    if int(sys.argv[3]) == 1:
        input_size = (256,512,1)
    inputs =Input(input_size)
    #conv0= (conv_base)(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    #conv11 = Flatten(input_shape = (1024,2048))(conv10)
    #conv12 = Dense(1, activation='sigmoid')(conv11)
    model = Model(input = inputs, output = conv10)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def OutputMaker(output):
    CleanOutput = output[0].split()
    Fix = np.zeros((350*4,525*4))
    for Y in range(0, int(len(CleanOutput)/2)):
        Overflow = 0
        for value in range(0,int(CleanOutput[2*Y+1])):
            position = int(CleanOutput[2*Y])%(350*4) + value - Overflow*350*4
            #print(position)
            if position >= 350*4:
                Overflow = Overflow + 1
                position = int(CleanOutput[2*Y])%(350*4) + value - Overflow*350*4
            Fix[position,int(int(CleanOutput[2*Y])/(350*4))+Overflow-1] = 1
    Fix = cv2.resize(Fix, (im_height, im_width))
    return(Fix) 

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#Set parameters for training:
SegNum = int(sys.argv[1]) #now many segmentations to have
Type = sys.argv[2] #type of flower for training
BGR = int(sys.argv[3]) # send black and white, BGR, mean_value,  or gaussian filter to training image

print(SegNum, Type, BGR)
model = unet()
model.compile(optimizer='adam', loss=dice_coef_loss, metrics = [dice_coef])
model.summary()

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(Type+sys.argv[1]+sys.argv[3]+'modelBackup.h5', verbose=1, save_best_only=True)

# Set some parameters
os.chdir('/depot/wwtung/data/Brittoa/Kaggle/understanding_cloud_organization')

im_width = 256
im_height = 512
images = glob.glob("train_images/*.jpg")

trainOutput = pd.read_csv("train.csv")
if BGR == 1:
    dimen = 1
else:
    dimen = 3
y = 0
while y < 55:
    x = 0
    X_train = np.zeros((5545, im_width, im_height, dimen), dtype=np.uint8)
    Labels = []
    while(x < 5545):
        imagePath = x
        if imagePath+100*y > 5545:
            break
        imageNum = imagePath
        print(imageNum+100*y, "of", len(images))
        img = load_img(images[imageNum])
        X = img_to_array(img)
        if SegNum > 1:
            X = seg.slic(X,n_segments=SegNum)
        if BGR == 0: #Original Image
            BGR = 0
        if BGR == 1:#Black and white image
            X = X[:,:,0]
        if BGR == 2:#Color from black and white
            X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        if BGR == 3:#Mean value
            mean_filter = np.ones((3,3))
            X = cv2.filter2D(X, -1, mean_filter)
            mx = np.max(np.abs(X))
            mn = np.min(np.abs(X))
            X= (mx - X) / (mx - mn)
        if BGR == 4:#Gaussian
            v = cv2.getGaussianKernel(5,10)
            gaussian = v*v.T
            X = cv2.filter2D(X, -1, gaussian)
            mx = np.max(np.abs(X))
            mn = np.min(np.abs(X))
            X= (mx - X) / (mx - mn)
        X = resize(X, (im_width,im_height, dimen), mode='constant', preserve_range=True)
        X_train[imageNum] = X
        Num = images[imageNum]
        imagechar = trainOutput[trainOutput['Image_Label']==(Num[13:]+'_'+Type)]#gather outputs
        output = list(imagechar['EncodedPixels'])
        if isinstance(output[0], float) != True:
            OutputClean = np.array(OutputMaker(output))#Unencode output values
            Labels.append(OutputClean.reshape(im_width*im_height))
            print(imageNum)
        else:
            Labels.append(np.zeros(im_width*im_height))
        x = x+1
    data = X_train/255
    print("scale")
    Labels = np.array(Labels)
    print("array Labels")
    data = data.reshape(-1, im_width,im_height, dimen)
    print("reshape 1")
    Labels = Labels.reshape(-1,im_width,im_height, 1)
    print("reshape 2")
    print("[INFO] training network...")
    results = model.fit(data,Labels, validation_split=0.1, batch_size=8, epochs=10, 
                     callbacks=[checkpointer])
    loss= results.history["loss"]
    loss = np.array(loss)
    np.savetxt(str(Type)+str(SegNum)+str(BGR)+"Model10.txt",loss,delimiter=',')
    model_json = model.to_json()
    with open(str(Type)+str(SegNum)+str(BGR)+"Model.json", 'w') as json_file:
        json_file.write(model_json)   
    model.save_weights(str(Type)+str(SegNum)+str(BGR)+"Model10.h5")
    print("Saved model")
    y = y + 1 