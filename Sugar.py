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

batch_size = 512
epochs = 15
num_classes = 183750
BatchInMemory = 640
Optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

Sugar_model = Sequential()
Sugar_model.add(Conv2D(128, kernel_size=(5, 5),activation='relu',input_shape=(350,525,1)))
Sugar_model.add(LeakyReLU(alpha=0.1))
Sugar_model.add(MaxPooling2D((2, 2),padding='same'))
Sugar_model.add(Conv2D(256, (5, 5), activation='relu'))
Sugar_model.add(LeakyReLU(alpha=0.1))
Sugar_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
Sugar_model.add(Conv2D(512, (3, 3), activation='relu'))
Sugar_model.add(LeakyReLU(alpha=0.1))                  
Sugar_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
Sugar_model.add(Flatten())
Sugar_model.add(Dense(1024, activation='relu'))
Sugar_model.add(LeakyReLU(alpha=0.1))                  
Sugar_model.add(Dense(num_classes, activation='relu'))
Sugar_model.compile(loss='mse', optimizer= Optimizer)

x = 0
print("[INFO] loading images...")
os.chdir('/depot/wwtung/data/Brittoa/Kaggle/understanding_cloud_organization')
trainOutput = pd.read_csv("train.csv")
images = glob.glob("train_images/*.jpg")

while BatchInMemory*x < len(images):
    data = []
    SuLabels = []
    for imagePath in range(BatchInMemory*x,BatchInMemory*x+BatchInMemory):
        if imagePath > 5545:
            break
        imageNum = imagePath
        print(imageNum+1, "of", len(images))
        dataInput =(np.array(cv2.imread(images[imageNum],0)))
        dataInput = cv2.resize(dataInput, (525, 350))
        data.append(np.array(dataInput))
        Num = images[imageNum]
        imagechar = trainOutput[trainOutput['Image_Label']==(Num[13:]+'_Sugar')]
        output = list(imagechar['EncodedPixels'])
        if isinstance(output[0], float) != True:
            OutputCleanSugar = np.array(OutputMaker(output))
            SuLabels.append(OutputCleanSugar.reshape(183750))
        else:
            SuLabels.append(np.zeros(183750))
    data = np.array(data) / 255 #Scales data
    SuLabels = np.array(SuLabels)
    train_XSu,valid_XSu,train_YSu,valid_YSu = train_test_split(data, SuLabels, 
                                                               test_size=0.2, 
                                                               random_state=13)
    train_XSu = train_XSu.reshape(-1, 350,525, 1)
    valid_XSu = valid_XSu.reshape(-1,350,525, 1)
    print("[INFO] training network...")
    Sugar_model.fit(train_XSu, train_YSu, 
                     batch_size=batch_size,epochs=epochs,verbose=1,validation_data=
                     (valid_XSu, valid_YSu))
    model_json = Sugar_model.to_json()
    with open("Sugarmodel5.json", "w") as json_file:
        json_file.write(model_json)   
    Sugar_model.save_weights("Sugarmodel5.h5")
    print("Saved model to disk 3")
    x = x+1