import tensorflow as tf
import keras
from keras.layers import Input,Dense,Convolution2D,MaxPooling2D,GlobalAveragePooling2D,LSTM
from keras.layers import Dropout, Permute, Reshape
from keras.datasets import cifar10
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model 
from keras import backend as k 
from keras.utils import np_utils
from tensorflow.python.ops.control_flow_ops import with_dependencies

import numpy as np
import math
import os
import argparse
import random
import sys
import time


parameter_servers = ["localhost:2222"]
#workers = [ "localhost:2223","localhost:2224","localhost:2225"]
workers = [ "localhost:2223","localhost:2224"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# Define input flags to identify the job and task
tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

#start a server for specific task
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#config.log_device_placement = True
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
server = tf.train.Server(cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config= config)

def prepare_input_data(X_train, X_test):
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return X_train, X_test

def prepare_output_data(y_train, y_test):
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return y_train, y_test

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
x_train, x_test = prepare_input_data(X_train[:10000], X_test[:1000])
y_train, y_test = prepare_output_data(Y_train[:10000], Y_test[:1000])
print(x_train.shape, x_test.shape)
# Network parameters
n_classes = 10 
image_size = 32
channel_size = 3
n_samples =  x_train.shape[0]
batch_size = 128
training_epochs = 5
num_iterations = n_samples//batch_size
test_step = 100
sigma = 1e-3
lr = 1e-2
input_shape = (image_size,image_size, channel_size)
frequency = 100

LOG_DIR = 'sync_logs'
print('parameters specification finished!')

def cnn():
    model = Sequential()
   
    model.add(Convolution2D(32,(3,3), strides = (1, 1),activation = 'relu', input_shape=input_shape)) 
    
    model.add(Convolution2D(64,(3,3), strides = (1, 1),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(256,(3,3), strides = (1, 1),activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))

    model.add(GlobalAveragePooling2D())    
    #model.add(Dense(1024, activation='relu'))
    #model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(10,activation ='softmax'))
    #RCNN
    model.add(Reshape((256,-1)))
    model.add(Permute((2,1)))
    model.add(LSTM(512, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation ='softmax'))

    model.summary()
    return model

def optimizer(model, targets, global_step):
    predictions = model.output
    starter_learning_rate = lr
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   10000, 0.96,staircase = True)       
    loss = tf.reduce_mean(keras.losses.categorical_crossentropy(targets, predictions))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                replicas_to_aggregate=2,
                                                total_num_replicas = 2)
    with tf.control_dependencies(model.updates):
        barrier = tf.no_op(name='update_barrier')
 
    with tf.control_dependencies([barrier]):
        # Compute gradients of `loss` for the variables. This is the first part of `minimize()`. 
        grads = optimizer.compute_gradients(loss,
                                            model.trainable_weights,
                                            colocate_gradients_with_ops=False)
       
         # Apply gradients to variables. This is the second part of `minimize()`.
        grad_updates = optimizer.apply_gradients(grads, global_step = global_step)

    train_op = with_dependencies([grad_updates],
                                 loss,
                                 name = 'train')
    tf.summary.scalar('learning_rate',learning_rate) 
    tf.summary.histogram('pred_y', predictions)
    tf.summary.histogram('gradients', train_op)
    
    #Option 2
    #with tf.control_dependencies([grad_updates]):
       #train_op =tf.identity( loss,
                               #name='train')
        
    correct_prediction = tf.equal(tf.argmax(targets,1), tf.argmax(predictions,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return (train_op,loss, predictions, accuracy, optimizer)

def train(train_op, loss, merged, global_step, accuracy):
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph = tf.get_default_graph())
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'test'),graph = tf.get_default_graph())
    
    start_time = time.time()
    for i in range(training_epochs*n_samples//batch_size):               
                offset = (i*batch_size) % n_samples
                batch_x = x_train[offset:(offset+batch_size),:,:,:]
                batch_y = y_train[offset:(offset+batch_size),:]

                summary, _,cost,acc,step = sess.run([merged, train_op,loss,accuracy,global_step],
                                                    feed_dict={model.inputs[0]: batch_x, targets: batch_y})
                train_writer.add_summary(summary,i)

                if i % frequency == 0:
                    elapsed_time = time.time() - start_time 
                    start_time = time.time()
                    print('Step: %d '% int(step),
                         "Epoch: %2d "% int(i),
                         'Cost: %.4f '% float(cost),
                         'Accuracy: %.4f ' % float(acc),
                         'AvgTime: %3.2fms'%float(elapsed_time*100/frequency))

    for i in range(test_step):
        if i % 10 == 0:
            summary, test_accuracy = sess.run([merged,accuracy],feed_dict={model.inputs[0]: x_test, targets: y_test})
            test_writer.add_summary(summary, i)
            print('Test accuracy at step %s: %s' % (i, test_accuracy)) 

print('FLAGS testing begin!')
if FLAGS.job_name == 'ps':
    server.join()
elif FLAGS.job_name == 'worker':
    print('Training begin!')
    is_chief = (FLAGS.task_index == 0) #checks if this is the chief node

    # Assign operations to local server
    with tf.device(tf.train.replica_device_setter(ps_tasks=1,
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        keras.backend.set_learning_phase(1)
        keras.backend.manual_variable_initialization(True)
        model = cnn()
        
        with tf.name_scope('input'):
            targets = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
              
        global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')    
        train_op,loss, predictions, accuracy, optimizer = optimizer(model, targets, global_step)
                    
        tf.summary.scalar('loss',loss) 
        tf.summary.scalar('accuracy',accuracy)
        tf.summary.image('input',model.inputs[0],10)

        merged = tf.summary.merge_all()    
        init_tokens_op = optimizer.get_init_tokens_op()
        #chief_queue_runner = optimizer.get_chief_queue_runner()     
        init_op = tf.global_variables_initializer()
    
    #Session
    sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
    stop_hook = tf.train.StopAtStepHook(last_step = 10000)
    chief_hooks = [sync_replicas_hook, stop_hook]
    scaff = tf.train.Scaffold(init_op = init_op)
    
    begin_time = time.time()
    print("Waiting for other servers")
    with tf.train.MonitoredTrainingSession(master = server.target,
                                          is_chief = is_chief,
                                          checkpoint_dir = LOG_DIR,
                                          scaffold = scaff,
                                          hooks = chief_hooks) as sess: 
       # tf.reset_default_graph()
        keras.backend.set_session(sess)
        print('Starting training on worker %d'%FLAGS.task_index)
        count = 0
        while not sess.should_stop() and count < 10000:
            train(train_op,loss, merged, global_step, accuracy)
            count += 1
 
    print("done")





