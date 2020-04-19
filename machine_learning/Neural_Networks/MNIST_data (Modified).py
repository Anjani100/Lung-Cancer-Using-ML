#!/usr/bin/env python
# coding: utf-8

# In[41]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
# 28 * 28 images of hand-written digit 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Unpacking the data
# print(x_train[0])

# plt.imshow(x_train[0])
# plt.imshow(x_train[0], cmap = plt.cm.binary) # To show the original greyscale image
# plt.show()

x_train = tf.keras.utils.normalize(x_train, axis = 1)  # This brings down the value of the tensors from a range of 0-255 to 0-1
x_test = tf.keras.utils.normalize(x_test, axis = 1)    # This provides a better accuracy for the model (40 % without normalization
                                                       #                                                97 % with normalization)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 5)
# Check for overfitment
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


# In[42]:


model.save('num_reader.model')


# In[43]:


new_model = tf.keras.models.load_model('num_reader.model')
predictions = new_model.predict([x_test])
print(predictions)


# In[50]:


print(np.argmax(predictions[100]))
plt.imshow(x_test[100], cmap = plt.cm.binary)
plt.show()


# In[ ]:




