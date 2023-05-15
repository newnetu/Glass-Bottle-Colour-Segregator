#!/usr/bin/env python
# coding: utf-8

# In[40]:


import cv2
import time
import serial
from PIL import Image
import numpy as np
from keras.models import load_model

# Connect to the Arduino Uno
ser = serial.Serial('COM5', 9600)

# Load the model
model_path = r'C:\Nathans Projects\EE297_Final\brown_clear_green_glass.h5'
model = load_model(model_path)

# Define the labels for each class
labels = ['brown', 'green', 'clear']
while True:
        
        # Capture an image from the webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        # Resize the image
        img = Image.fromarray(frame)
        img = img.resize((256, 256))
        # Convert the image to a numpy array
        x = np.array(img)
        # Expand the dimensions of the array to match the model's input shape
        x = np.expand_dims(x, axis=0)
        # Make the prediction
        prediction = model.predict(x)

        # Get the predicted class label
        predicted_class = labels[np.argmax(prediction)]
        

        # Send a signal to the Arduino based on the prediction
        if predicted_class == 'brown':
            # Brown bottle detected, activate the actuator and flash the brown LED
            ser.write(b'1')
            time.sleep(1.5)
            ser.write(b'4') # stop the actuator
            print("Brown bottle detected with probability", prediction)
        elif predicted_class == 'green':
            # Green bottle detected, activate the actuator and flash the green LED
            ser.write(b'2')
            time.sleep(2.5)
            ser.write(b'4') # stop the actuator
            print("Green bottle detected with probability", prediction)
        else:
            # Clear bottle detected#
                ser.write(b'3')
                print("Clear bottle detected with probability", prediction)

