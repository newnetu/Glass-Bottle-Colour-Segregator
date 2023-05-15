#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Tensorflow and 
import tensorflow as tf
import os


# In[2]:


import cv2
import imghdr


# In[3]:


data_dir = r'C:\Nathans Projects\EE297_second\data'


# In[4]:


image_exts = ['jpeg','jpg', 'bmp', 'png']


# In[5]:


for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            os.remove(image_path)


# In[6]:


import numpy as np
from matplotlib import pyplot as plt


# In[7]:


data = tf.keras.utils.image_dataset_from_directory(data_dir)


# In[8]:


data_iterator = data.as_numpy_iterator()


# In[9]:


batch = data_iterator.next()


# In[10]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# In[11]:


data = data.map(lambda x,y: (x/255, y))


# In[14]:


import os
import shutil
import random

# Set the path to your data directory
data_dir = r'C:\Nathans Projects\EE297_second\data'

# Set the percentages for the train/val split
train_pct = 0.8
val_pct = 0.2

# Create directories for train and validation data
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Loop over each class directory and split the images into train and val sets
for class_name in ['brown_glass', 'green_glass', 'clear_glass']:
    # Set the path to the class directory
    class_dir = os.path.join(data_dir, class_name)
    
    # Get a list of all the image filenames in the class directory
    image_filenames = os.listdir(class_dir)
    
    # Shuffle the filenames to ensure a random split
    random.shuffle(image_filenames)
    
    # Determine the split index for train/val
    split_idx = int(len(image_filenames) * train_pct)
    
    # Split the image filenames into train and validation sets
    train_filenames = image_filenames[:split_idx]
    val_filenames = image_filenames[split_idx:]
    
    # Create subdirectories for each class in the train and val directories
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    
    # Copy the train images to the train subdirectory for this class
    for filename in train_filenames:
        src_path = os.path.join(class_dir, filename)
        dst_path = os.path.join(train_class_dir, filename)
        shutil.copy(src_path, dst_path)
    
    # Copy the validation images to the validation subdirectory for this class
    for filename in val_filenames:
        src_path = os.path.join(class_dir, filename)
        dst_path = os.path.join(val_class_dir, filename)
        shutil.copy(src_path, dst_path)


# In[24]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = r'C:\Nathans Projects\EE297_Final\data\train'
val_dir = r'C:\Nathans Projects\EE297_Final\data\val'

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(256, 256),batch_size=32,class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_dir,target_size=(256, 256),batch_size=32,class_mode='categorical')

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train_generator,epochs=20,validation_data=val_generator, callbacks=[tensorboard_callback])


# In[25]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[26]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[27]:


import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


# Load the image from file
img_path =  r'C:\Users\natew\Downloads\20200624_102652770_iOS-scaled.webp'
img = image.load_img(img_path, target_size=(256, 256))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Normalize the image
img /= 255.0

# Make a prediction with the model
prediction = model.predict(img)
print(prediction)

# Print the predicted class
if prediction[0][2] > 0.33:
    print('The bottle is green with probability {:.2f}%'.format(prediction[0][2]*100))
elif prediction[0][0] >0.33:
    print('The bottle is brown with probability {:.2f}%'.format(prediction[0][0]*100))
elif prediction[0][1] > 0.33: 
    print('The bottle is clear with probability {:.2f}%'.format(prediction[0][1]*100))


# In[28]:


from tensorflow.keras.models import load_model


# In[29]:


model.save(os.path.join(r'C:\Nathans Projects\EE297_second','brown_clear_green_glass.h5'))


# In[ ]:




