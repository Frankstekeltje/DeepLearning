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

#clearing dataset from corrupted images
num_passed = 0
for images in ("frank", "caroline"):
    folder_path = os.path.join('C:\FaceDetection\DeepLearning\images', images)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jpg = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jpg:
            num_passed += 1
            #Delete corrupted images
            os.remove(fpath)

print("Deleted items %d" % num_passed)

image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
"C:\FaceDetection\DeepLearning\images",
labels="inferred",
validation_split=0.2,
subset="training",
seed=1337,
image_size=image_size,
batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
"C:\FaceDetection\DeepLearning\images",
validation_split=0.2,
subset="validation",
seed=1337,
image_size=image_size,
batch_size=batch_size,
)

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
    plt.show()


data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])



plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
    plt.show()

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

def create_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    layers.Rescaling(1.0/255)(x)

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


# creating model
model = create_model((180,180,3), 2)
#optimizing model
compile_model(model, 'sparse_categorical_crossentropy', 'adam')



# CAMERA can be 0 or 1 based on default camera of your computer.
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# camera = cv2.imread("c:/faceDetection/DeepLearning/images/frank/20.jpg")
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

# Grab the labels from the labels.txt file. This will be used later.
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('frank.h5')
# print(model)

# #training model
model.fit(train_ds, validation_data = val_ds, batch_size = 150, epochs = 80)
model.save("frank.h5") #model will be save in root folder to be later called out for prediction

while True:
    img = cv2.imread("c:/faceDetection/DeepLearning/images/frank/20.jpg") #loading input image
    cv2.imshow('image',img)
    keyboard_input = cv2.waitKey(1)
    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break


img2 = keras.preprocessing.image.load_img(
    "c:/faceDetection/DeepLearning/images/frank/20.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img2)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
print(predictions)
score = np.argmax(predictions)
print(score)


print(
    "This image is %.2f percent caroline and %.2f percent frank."
    % (100 * (1 - score), 100 * score)
)

def get_className(classNo):
    if classNo == 0:
        return "Frank"
    elif classNo == 1:
        return "Caroline"

while True:
    # Grab the webcameras image.
    ret, frame = camera.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        crop_image = frame
        # Resize the raw image into (224-height,224-width) pixels.
        image = cv2.resize(crop_image, (180, 180), interpolation=cv2.INTER_AREA)
        image = image.reshape(1, 180, 180, 3)
        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 180, 180, 3)
        # Normalize the image array
        image = (image / 127.5) - 1
        # Have the model predict what the current image is. Model.predict
        # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
        # it is the first label and 80% sure its the second label.
        probabilities = model.predict(image)
        # Print what the highest value probabilitie label
        print(probabilities)
        classIndex = np.argmax(probabilities, axis=-1)
        print(classIndex)
        probabilityValue = np.amax(probabilities)
        if classIndex == 0:
            print("test")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -2)
            cv2.putText(frame, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        elif classIndex == 1:
            print("hallo")
            cv2.rectangle(frame, (x, y), (x+w,y+h),(0,255,0),2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255,0),-2)
            cv2.putText(frame, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255,255),1, cv2.LINE_AA)
        cv2.putText(frame, str(round(probabilityValue*100, 2))+"%" , (180, 75), font, 0.75, (255, 0,0),2, cv2.LINE_AA)
        # Listen to the keyboard for presses.
        # Show the image in a window
    cv2.imshow('Frame', frame)
    keyboard_input = cv2.waitKey(1)
    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
