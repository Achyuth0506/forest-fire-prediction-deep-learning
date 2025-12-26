from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2
from selenium import webdriver

def prep(data_folder_path):
    dirs = os.listdir(data_folder_path)
    images = []
    labels = []
    
    for dir_name in dirs:
        print(dir_name)
        if dir_name=='0': 
            label = 0
        elif dir_name=='1':
            label = 1
        
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            images.append(image_path)
            labels.append(label)
    
    return images, labels

# Appending Images
path = "/content/drive/My Drive/archive/Train"
train_img, train_lab = prep(path)

path = "/content/drive/My Drive/archive/test"
test_img, test_lab = prep(path)

tl = len(train_img)
train_data = []
c=0
for path in train_img:
    img = load_img(path, target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    train_data.append(img)
    c+=1
    print(c)

tl = len(test_img)
test_data = []
c=0
for path in test_img:
    img = load_img(path, target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    test_data.append(img)
    c+=1
    print(c)

trainY, testY = np.array(train_data), np.array(test_data)
trainX, testX = np.array(train_lab), np.array(test_lab)

# Changing as an Array
print(trainY.shape)

import pickle
with open("fire.pickle","wb") as f:
    pickle.dump([trainY, trainX, testY, testX], f)

# Saving Preprocessed Images
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Data Augmentation
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                       input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

# Layer of Neural Network
INIT_LR = 1e-4
EPOCHS = 100
BS = 64

opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

from tensorflow import keras

class callbacks(keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.86:
            print("\n Accuracy reached \n")
            self.model.stop_training = True

callbacks = callbacks()

H = model.fit(
    aug.flow(trainY, trainX),
    steps_per_epoch=len(trainX)//BS,
    validation_data=(testY, testX),
    validation_steps=len(testX)//BS,
    epochs=EPOCHS,
    shuffle=True,
    callbacks=[callbacks])

# Training Results
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
loss = H.history['loss']
val_loss = H.history['val_loss']
epochs = range(len(acc))

plt.figure(figsize=(10,6))
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)

# TESTING AND TRAINING LOSS FUNCTION ACCURACY
# PLOTTING TRAINING AND TESTING ACCURACY AND LOSS FUNCTION

from keras.models import model_from_json
model_json = model.to_json()
with open("model_garbage_arch.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_garbage_weights.h5")

# SAVING AND LOADING TRAINED MODEL
from tensorflow.keras.models import model_from_json
json_file = open('/content/drive/My Drive/archive/model_garbage_arch.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("/content/drive/MyDrive/archive/model_garbage_weights.h5")