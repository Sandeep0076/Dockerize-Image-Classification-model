#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import regularizers
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16
from sklearn import preprocessing
from sklearn.decomposition import PCA
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


SIZE = 256  #Resize images
# Capture train data and labels into respective lists
train_images = []
train_labels = [] 

for directory_path in glob.glob("augmented_img/train/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.*")):
      try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)
      except:
        break

#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Capture test/validation data and labels into respective lists
test_images = []
test_labels = [] 
for directory_path in glob.glob("augmented_img/validate/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.*")):
      try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)
      except:
        break
        

#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Encode labels from text to integers.
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)


#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train = train_images
del train_images
y_train = train_labels_encoded
del train_labels_encoded

x_test = test_images
del test_images
y_test = test_labels_encoded
del test_labels_encoded

###################################################################
# Scale pixel values to between 0 and 1
x_train = x_train / 255.0

x_test =  x_test / 255.0

#One hot encode y values for neural network. Not needed for Random Forest
from tensorflow.keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


#############################
#Load VGG model with imagenet trained weights and without classifier/fully connected layers
#We will use this as feature extractor. 
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False

#Now, let us extract features using VGG imagenet weights
#Train features
train_feature_extractor=VGG_model.predict(x_train)
train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0], -1)
#test features
test_feature_extractor=VGG_model.predict(x_test)
test_features = test_feature_extractor.reshape(test_feature_extractor.shape[0], -1)


#Pick the optimal number of components. This is how many features we will have 
#for our machine learning
n_PCA_components = 1500
pca = PCA(n_components=n_PCA_components)
train_PCA = pca.fit_transform(train_features)
test_PCA = pca.transform(test_features) #Make sure you are just transforming, not fitting. 

def create_model():
    model2 = Sequential()
    model2.add(Flatten(input_shape=(n_PCA_components,)))
    model2.add(Dense(128,activation='relu', kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.1)))
    model2.add(Dropout(0.5))
    model2.add(Dense(4, activation='softmax'))

    # compile the model
    model2.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
    K.set_value(model2.optimizer.learning_rate, 0.0001)
    
    return model2



chk = ModelCheckpoint("myModel.h5", monitor='val_loss', save_best_only=True, mode='min', save_weights_only=True) 
callbacks_list = [chk]
#pass callback on fit
# train model using features generated from VGG16 model
model2 = create_model()
hist = model2.fit(train_PCA, y_train_one_hot, epochs=20, batch_size=5, validation_data=(test_PCA, y_test_one_hot),callbacks=callbacks_list)

