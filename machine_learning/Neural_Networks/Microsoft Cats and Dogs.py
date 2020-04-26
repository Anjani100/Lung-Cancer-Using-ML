#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import os, cv2

DATADIR = "D:\\Dataset\\Microsoft Cats and Dogs\\PetImages"
CATEGORIES = ["Dog", "Cat"]
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                #plt.imshow(img_array, cmap = 'gray')
                #plt.show()
                new_array = cv2.resize(img_array, (50, 50))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()


# In[9]:


print(len(training_data))


# In[12]:


import random
random.shuffle(training_data)

# for sample in training_data:
#     print(sample[1])


# In[14]:


X = []
y = []
IMG_SIZE = 50

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[15]:


import pickle
pickle_out_X = open('X.pickle', 'wb')
pickle.dump(X, pickle_out_X)
pickle_out_X.close()

pickle_out_y = open('y.pickle', 'wb')
pickle.dump(y, pickle_out_y)
pickle_out_y.close()


# In[16]:


import pickle

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
print(X[1])


# In[ ]:




