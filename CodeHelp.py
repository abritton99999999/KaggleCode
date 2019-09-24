# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import glob
import pandas as pd

def OutputMaker(output):
    output = output.split()
    CleanOutput = [0]*(len(output))
    for num in range(0, len(output)):
        CleanOutput[num] = int(output[num])
    data = [0]*350*550
    x=0
    while x <= len(CleanOutput)/2:
        raa = list(range(round(CleanOutput[x]/16), round(CleanOutput[x]/16.)+round(CleanOutput[x+1]/4.)))
        for number in raa:
            data[number] = 1
            x=x+2
    return(CleanOutput)
    
# initialize the data and labels
print("[INFO] loading images...")
data = []
#types = "Sugar", "Fish", "Flower", "Gravel"
types = [0,255,255*2,255*3]
labels = []
os.chdir('/depot/wwtung/data/Brittoa/Kaggle/understanding_cloud_organization')
trainOutput = pd.read_csv("train.csv")
images = glob.glob("train_images/*.jpg")
model = Sequential()
model.add(Dense(100000, input_shape=(192500,), activation="sigmoid"))
model.add(Dense(50000, activation="sigmoid"))
model.add(Dense(192500, activation="softmax"))
INIT_LR = 0.01
EPOCHS = 75
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# evaluate the network
# loop over the input images
x = 0
while x <= 3:
    data = []
    labels = []
    for imagePath in range(0,int(len(images)/4)):
        print(imagePath, "of", len(images))
        imagechar = trainOutput.loc[imagePath*4] #Gets image data
        imageNameFull = imagechar[0] #finds name and type of image
        imageName= imageNameFull[0:7] #separates name
        imageFile = images[images == imageName] #finds image file for name
        image = cv2.imread(imageFile, 0)
        #print(image.shape)
        for type in range(0,len(types)):
            dataData = cv2.resize(image, (350, 550)).flatten() #gets final data
            #print(dataData.shape)
            dataData[0] = (types[type])
            dataData = np.array(dataData, dtype="float") / 255.0
            #data.append(dataData)
            #print(len(data))
            output = imagechar[1]
            if isinstance(output, list) == True:
                labels = (Outputmaker(output))
            else:
                labels = [0]*350*550
            dataData = np.asarray(dataData)
            dataData = dataData.reshape(1,-1)
            labels = np.asarray(labels)
            labels = labels.reshape(1,-1)
            H = model.fit(dataData, labels,epochs=EPOCHS)
    #data = np.array(data, dtype="float") / 255.0
    #(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.2, random_state=42)
    print("[INFO] training network...")
    x=x +1
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1)))

 
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()