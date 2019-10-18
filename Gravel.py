# import the necessary packages
#from sklearn.metrics import classification_report
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
from tensorflow.keras import layers
from keras.models import model_from_json


def OutputMaker(output):
    output = output[0].split()
    CleanOutput = output
    Fix = np.zeros((350*4,525*4))
    for Y in range(0, int(len(CleanOutput)/2)):
        #print(Y, len(CleanOutput)/2)
        #print(CleanOutput[2*Y], CleanOutput[2*Y+1])
        #print(int(CleanOutput[2*Y]/350))
        #print(CleanOutput[2*Y]%350)
        Overflow = 0
        for value in range(0,int(CleanOutput[2*Y+1])):
            position = int(CleanOutput[2*Y])%(350*4) + value - Overflow*350*4
            #print(position)
            if position >= 350*4:
                Overflow = Overflow + 1
                position = int(CleanOutput[2*Y])%(350*4) + value - Overflow*350*4
            #print(position,int(CleanOutput[2*Y]/350)+Overflow-1)
            Fix[position,int(int(CleanOutput[2*Y])/(350*4))+Overflow-1] = 1
    Fix = cv2.resize(Fix, (525, 350))
    return(Fix)

batch_size = 64
epochs = 10
num_classes = 183750
BatchInMemory = 80
Optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

Gravel_model = Sequential()
Gravel_model.add(Conv2D(128, kernel_size=(5, 5),activation='relu',input_shape=(350,525,1)))
Gravel_model.add(LeakyReLU(alpha=0.1))
Gravel_model.add(MaxPooling2D((2, 2),padding='same'))
Gravel_model.add(Conv2D(256, (5, 5), activation='relu'))
Gravel_model.add(LeakyReLU(alpha=0.1))
Gravel_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
Gravel_model.add(Conv2D(512, (3, 3), activation='relu'))
Gravel_model.add(LeakyReLU(alpha=0.1))                  
Gravel_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
Gravel_model.add(Flatten())
Gravel_model.add(Dense(1024, activation='relu'))
Gravel_model.add(LeakyReLU(alpha=0.1))                  
Gravel_model.add(Dense(num_classes, activation='relu'))
Gravel_model.compile(loss='mse', optimizer= Optimizer)

x = 0
print("[INFO] loading images...")
os.chdir('/depot/wwtung/data/Brittoa/Kaggle/understanding_cloud_organization')
trainOutput = pd.read_csv("train.csv")
images = glob.glob("train_images/*.jpg")

while BatchInMemory*x < len(images):
    data = []
    GrLabels = []
    for imagePath in range(BatchInMemory*x,BatchInMemory*x+BatchInMemory):
        if imagePath > 5545:
            break
        imageNum = imagePath
        print(imageNum, "of", len(images))
        dataInput =(np.array(cv2.imread(images[imageNum],0)))
        dataInput = cv2.resize(dataInput, (525, 350))
        data.append(np.array(dataInput))
        Num = images[imageNum]
        imagechar = trainOutput[trainOutput['Image_Label']==(Num[13:]+'_Gravel')]
        output = list(imagechar['EncodedPixels'])
        if isinstance(output[0], float) != True:
            OutputCleanGravel = np.array(OutputMaker(output))
            GrLabels.append(OutputCleanGravel.reshape(183750))
        else:
            GrLabels.append(np.zeros(183750))
    data = np.array(data) / 255 #Scales data
    GrLabels = np.array(GrLabels)
    train_XGr,valid_XGr,train_YGr,valid_YGr = train_test_split(data, GrLabels, 
                                                               test_size=0.2, 
                                                               random_state=13)
    train_XGr = train_XGr.reshape(-1, 350,525, 1)
    valid_XGr = valid_XGr.reshape(-1,350,525, 1)
    print("[INFO] training network...")
    Gravel_model.fit(train_XGr, train_YGr, 
                     batch_size=batch_size,epochs=epochs,verbose=1,validation_data=
                     (valid_XGr, valid_YGr))
    model_json = Gravel_model.to_json()
    with open("Gravelmodel6.json", "w") as json_file:
        json_file.write(model_json)   
    Gravel_model.save_weights("Gravelmodel6.h5")
    print("Saved model to disk 3")
    x = x+1