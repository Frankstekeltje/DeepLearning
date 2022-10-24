import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from keras.models import load_model
import numpy as np

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the model
model = load_model('keras_model.h5')

# CAMERA can be 0 or 1 based on default camera of your computer.
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

# Grab the labels from the labels.txt file. This will be used later.
labels = open('labels.txt', 'r').readlines()
font = cv2.FONT_HERSHEY_COMPLEX

def get_className(classNo):
    if classNo == 0:
        return "Frank"
    elif classNo == 1:
        return "Caroline"

while True:
    # Grab the webcameras image.
    ret, frame = camera.read()
    faces = faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        crop_image = frame
        # Resize the raw image into (224-height,224-width) pixels.
        image = cv2.resize(crop_image, (224, 224), interpolation=cv2.INTER_AREA)
        image = image.reshape(1, 224, 224, 3)
        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        # Normalize the image array
        image = (image / 127.5) - 1
        # Have the model predict what the current image is. Model.predict
        # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
        # it is the first label and 80% sure its the second label.
        probabilities = model.predict(image)
        print(probabilities)
        # Print what the highest value probabilitie label
        classIndex = np.argmax(probabilities)
        probabilityValue = np.amax(probabilities)
        if classIndex == 0:
            print("test")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -2)
            cv2.putText(frame, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        elif classIndex == 1:
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
