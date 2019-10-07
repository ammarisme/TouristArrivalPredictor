import pandas as pd
import pymongo
import numpy as np
import json
import matplotlib.pyplot as pyplot

from sklearn.preprocessing import MinMaxScaler

periods = 3

client = pymongo.MongoClient("localhost", 27017)
db = client.tourdb

placeCollection = db.places #get place table

placeData = placeCollection.find()

for place in placeData:
    
    tidx = pd.date_range('2019-09-01', periods=periods, freq='MS')
    #                     ^                                   ^
    #                     |                                   |
    #                 Start Date        Frequency Code for Month Start
    data = np.random.randint(low=75, high=place["max_count"], size =periods)
    
    series = pd.Series(data=data, index=tidx)
    series = pd.DataFrame(series,columns=['no_of_tourists'])
    
    print(series)
    
    pyplot.show()
    
    test_set = series
    
    test_set
    
    # Scale the Data
    scaler = MinMaxScaler()
    test_scaled = scaler.fit_transform(test_set)
    
    
    import tensorflow as tf
    
    tf.reset_default_graph()
    #  feature, the time series
    num_inputs = 1
    # Num of steps in each batch
    num_time_steps = periods
    #  neuron layer
    num_neurons = 400
    # Just one output, predicted time series
    num_outputs = 1
    
    ## increasing iterations, but decreasing learning rate
    learning_rate = 0.001 
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
    
    # Initialize the global variables
    init = tf.global_variables_initializer()
    
    # Create an instance of tf.train.Saver()
    saver = tf.train.Saver()
        
    # Predicting Future 
    test_set
    
    # Generative Session
    with tf.Session() as sess:
        # Use your Saver instance to restore your saved rnn time series model
        saver.restore(sess, "./model/ex_time_series_model")
    
        # Create a numpy array for your genreative seed from the last 12 months of the 
        # training set data. Hint: Just use tail(12) and then pass it to an np.array
        train_seed = list(test_scaled)
            
        ## Now create a for loop that 
        for iteration in range(periods):
            X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
            train_seed.append(y_pred[0, -1, 0])
    
    train_seed
    results = scaler.inverse_transform(np.array(train_seed[periods:]).reshape(periods,1))
    test_set['no_of_tourists'] = results
    
    test_set.index.name = 'date'
    series = pd.DataFrame(test_set)
    
    print(series)
    upsampled = series.resample('D')#day
    
    interpolated = upsampled.interpolate(method='spline', order=1)
    #interpolated.index = interpolated.index.astype('datetime64[ns]')
    
    
    interpolated.no_of_tourists = interpolated.no_of_tourists.astype(int)
    #print(interpolated)
    
    interpolated.to_csv(r'./predictions/place_' + str(place["_id"]) + '.csv', header=True)
    
    interpolated["date"] = interpolated.index.astype(str)
    
    interpolated = pd.DataFrame.from_dict(interpolated) #create list from dataframe
    interpolated = list(json.loads(interpolated.T.to_json()).values()) #convert to json format
    
    placeCollection.update_one({'_id' : place["_id"]}, {"$set" : {"predictions" : interpolated}})
    
    

