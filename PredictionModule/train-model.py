# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:21:51 2019

"""

import pandas as pd
import pymongo
import numpy as np
from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


client = pymongo.MongoClient("localhost", 27017)
db = client.tourdb

placeId = 21 # place id

dataCollection = db.dataset # get dataset table
placeCollection = db.places #get place table

dataset = list(dataCollection.find({'place_id': placeId}))

def parser(x):
	return pd.datetime.strptime(x, '%Y-%m')

key=0

#print(dataset)

for data in dataset:
    dataset[key]["no_of_tourists"] = int(dataset[key]["no_of_tourists"])
    dataset[key]["date"] = parser(str(int(data["year"])) + "-" + str(int(data["month"])))
    key=key+1
   
series = pd.DataFrame(dataset,columns=['date', 'no_of_tourists']).reset_index(drop=True).set_index('date')

#print(series)
#upsampled = series.resample('D')#day



#print(upsampled)

#interpolated = upsampled.interpolate(method='spline', order=2)
#print(interpolated.head(32))

print(len(series))
#print(count)

series.plot()
pyplot.show()

train_set = series[:-24]
test_set = series.tail(24)

#train_set, test_set = train_test_split(series, train_size=0.75, test_size=0.25, shuffle=False)
print("Train and Test size", len(train_set), len(test_set))


# Scale the Data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)

def next_batch(training_data,batch_size,steps):
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0,len(training_data)-steps) 

    # Create Y data for time series in the batches
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)

    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1) 

# Setting Up The RNN Model

import tensorflow as tf

tf.reset_default_graph()

#  feature, the time series
num_inputs = 1
# Num of steps in each batch
num_time_steps = 12
#  neuron layer
num_neurons = 400
# Just one output, predicted time series
num_outputs = 1

## increasing iterations, but decreasing learning rate
learning_rate = 0.0001 
# how many iterations to go through (training steps)
num_train_iterations = 20000
# Size of the batch of data
batch_size = 12

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# create the RNN Layer
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs) 

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# Loss Function and Optimizer
loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

#tf.reset_default_graph()

# Initialize the global variables
init = tf.global_variables_initializer()

# Create an instance of tf.train.Saver()
saver = tf.train.Saver()


# Session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        
        X_batch, y_batch = next_batch(train_scaled,batch_size,num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})
        
        if iteration % 100 == 0:
            
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    
    # Save Model for Later
    saver.save(sess, "./model/ex_time_series_model")
    
# Predicting Future
test_set

# Generative Session
with tf.Session() as sess:
    
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./model/ex_time_series_model")

    print(train_scaled[-24:])
    # Create a numpy array for your genreative seed from the last 12 months of the 
    # training set data. Hint: Just use tail(12) and then pass it to an np.array
    train_seed = list(train_scaled[-24:])
    
    print(train_seed)
    
    ## Now create a for loop that 
    for iteration in range(24):
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        train_seed.append(y_pred[0, -1, 0])


train_seed
results = scaler.inverse_transform(np.array(train_seed[24:]).reshape(24,1))
test_set['predicted'] = results
test_set

test_set.plot()


