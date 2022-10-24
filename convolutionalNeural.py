# importing libraries
from tkinter import Frame
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import keras.layers as layers
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten

#loading dataset
(train_X, train_y), (val_X, val_y) = mnist.load_data()

# normalizing the dataset
train_X, val_X = train_X/255, val_X/255

# visualizing 9 rndom digits from the dataset
for i in range(331,340):
    plt.subplot(i)
    a = np.random.randint(0, train_X.shape[0], 1)
    plt.imshow(train_X[a[0]], cmap = plt.get_cmap('binary'))

plt.tight_layout()
plt.show()

def create_model(input_shape, num_classes):
    model = keras.Sequential([   
    layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = input_shape),
    layers.MaxPool2D(pool_size = 2),
    
    layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same'),
    layers.MaxPool2D(pool_size = 2),
    
    layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu', padding = 'same'),
    layers.MaxPool2D(pool_size = 2),
    
    layers.Flatten(),
    layers.Dense(units = 54, activation = 'relu'),
    layers.Dense(units = num_classes, activation = 'softmax')])
    
    return model

def compile_model(model, loss, optimizer='adam'):
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    model.summary()

def fitting_model(model, x, y, epoch):
    model.fit(x,y, shuffle = True, epochs = epoch)

# reshaping the independant variables
train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
val_X = val_X .reshape(val_X.shape[0], 28, 28, 1)

#encoding the dependant variable
train_y = np.eye(10)[train_y]
val_y = np.eye(10)[val_y]

# #creating model
# model = create_model((28,28,1))
# #optimizing model
# compile_model(model, 'adam', 'categorical_crossentropy')

# #training model
# history = model.fit(train_X, train_y, validation_data = (val_X, val_y), batch_size = 150, epochs = 80)
# model.save("cnn_digitclass.model") #model will be save in root folder to be later called out for prediction

#model performance visualization
# f = plt.figure(figsize=(20,8))

#accuracy
# plt1 = f.add_subplot(121)
# plt1.plot(history.history['accuracy'], label = str('Training accuracy'))
# plt1.plot(history.history['val_accuracy'], label = str('Validation accuracy'))
# plt.legend()
# plt.title('accuracy')

#loss
# plt2 = f.add_subplot(122)
# plt2.plot(history.history['loss'], label = str('Training loss'))
# plt2.plot(history.history['val_loss'], label = str('Validation loss'))
# plt.legend()
# plt.title('loss')

plt.show()
while True:
    img = cv2.imread("sample_img3.png") #loading input image
    img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA) #resizing to input shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #chaging to grayscale format
    img = cv2.bitwise_not(img) #the color scale was inverted, correcting inverted color scale
    img = cv2.Canny(img, 50, 50) # removing noise
    cv2.imshow('image',img)
    keyboard_input = cv2.waitKey(1)
    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break


predict_data = np.array([img])/255 #changing image data to array
predict_data = predict_data.reshape(1,28, 28, 1) #reshaping to input shape

# predicting the input 
from keras import models
model = models.load_model('cnn_digitclass.model') #loading pre-savedd model
prediction = model.predict(predict_data) #gives array
predicted_number = np.argmax(prediction) #extracts predicted number
print(predicted_number)