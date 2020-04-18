#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Extracting Input Data

mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

'''one_hot denotes that any one of the elements in the matrix (dimension -> number of nodes in output layer) will be 1.

1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
3 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
...
9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

'''

# Steps

'''
input_data > weights > hidden layer 1 (activation function) > weights > 
                       hidden layer 2 (activation function) > weights >
                       hidden layer 3 (activation function) > weights >
                       output_layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer) 

backpropagation

feed forward + backpropagation = epoch (total 10-15 will be sufficient for this dataset)

'''

# Defining nodes for each layer

n_nodes_hl = [784, 1500, 1500, 500, 10]

batch_size = 100

x = tf.placeholder('float', shape = (None, 784))
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_layer = []
    for i in range(len(n_nodes_hl) - 1):
        hidden_layer.append({'weights': tf.Variable(tf.random_normal([n_nodes_hl[i], n_nodes_hl[i + 1]])),
                                'biases': tf.Variable(tf.random_normal([n_nodes_hl[i + 1]]))})
    
    layers = [data]
    for i in range(len(hidden_layer) - 1):
        # hidden_layer * weights + biases
        l = tf.add(tf.matmul(layers[i], hidden_layer[i]['weights']), hidden_layer[i]['biases'])
        layers.append(tf.nn.relu(l))
    
    output = tf.matmul(layers[-1], hidden_layer[-1]['weights']) + hidden_layer[-1]['biases']
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    
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




