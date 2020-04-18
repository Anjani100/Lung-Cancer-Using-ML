#!/usr/bin/env python
# coding: utf-8

# In[24]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def maxPool2D(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def convolutional_neural_network(x):
    
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               #'W_conv3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
               'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}
    
    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              #'b_conv3': tf.Variable(tf.random_normal([128])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}
    
    # Reshaping the input to a 4D-Convolutional Layer
    x = tf.reshape(x, shape = [-1, 28, 28, 1])
    # 1st Convolution Layer
    conv1 = tf.nn.relu(conv2D(x, weights['W_conv1']) + biases['b_conv1'])
    # Max Pooling Layer (Down Sampling)
    conv1 = maxPool2D(conv1)
    
    # 2nd Convolution Layer
    conv2 = tf.nn.relu(conv2D(conv1, weights['W_conv2']) + biases['b_conv2'])
    # Max Pooling Layer (Down Sampling)
    conv2 = maxPool2D(conv2)
    
    # 3rd Convolution Layer
    #conv3 = tf.nn.relu(conv2D(conv2, weights['W_conv3']) + biases['b_conv3'])
    # Max Pooling Layer (Down Sampling)
    #conv3 = maxPool2D(conv3)
    
    # Fully Connected Layer
    # Reshape conv3 output to fit fully connected layer
    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    
    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # cycles for feed forward and backprop
    hm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
                
            print('Epoch', epoch + 1, 'completed out of a total of', hm_epochs, 'Loss:', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)


# In[ ]:




