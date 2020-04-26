#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os, cv2, random
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

DATADIR = "D:\\Dataset\\Microsoft Cats and Dogs\\PetImages"
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE = 200
training_data = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img = os.path.join(path, img)
            img_array = cv2.imread(img)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass

random.shuffle(training_data)
print(len(training_data))


# In[ ]:


X = []
y = []

for feature, label in training_data:
    X.append(feature)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)


# In[ ]:


pickle_out_X = open('X.pickle', 'wb')
pickle.dump(X, pickle_out_X)
pickle_out_X.close()

pickle_out_y = open('y.pickle', 'wb')
pickle.dump(y, pickle_out_y)
pickle_out_y.close()


# In[ ]:


# VGG-2 model for the dogs vs cats dataset
import sys
import pickle
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('y.pickle', 'rb')
y = pickle.load(pickle_in)
pickle_in.close()

X = X / 255.0

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = define_model()
model.fit(X, y, batch_size = 32, epochs = 6, validation_split = 0.1, shuffle = True) # 6 epochs is more than enough


# In[ ]:


# VGG-3 model for the dogs vs cats dataset
import sys
import pickle
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('y.pickle', 'rb')
y = pickle.load(pickle_in)
pickle_in.close()

X = X / 255.0

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = define_model()
model.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.1, shuffle = True)


# In[ ]:


# VGG-3 model for the dogs vs cats dataset
# Dropouts added
import sys
import pickle
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('y.pickle', 'rb')
y = pickle.load(pickle_in)
pickle_in.close()

X = X / 255.0

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = define_model()
model.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.1, shuffle = True)


# In[3]:


# vgg16 model used for transfer learning on the dogs and cats dataset
import sys
import pickle
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('y.pickle', 'rb')
y = pickle.load(pickle_in)
pickle_in.close()

X = X / 255.0

# define cnn model
def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=(200, 200, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = define_model()
model.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.1, shuffle = True)


# In[ ]:




