# coding: utf-8
import tensorflow as tf
import numpy as np
from mlxtend.preprocessing import one_hot
import argparse
import random
import sys
import time
from utils import mnist_reader
from tensorflow.contrib import slim
import os

parameter_servers = ["localhost:2222"]
workers = [ "localhost:2223", "localhost:2224",'localhost: 2225']
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# Input Flags
tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

 #Set up server
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement=True 
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
server = tf.train.Server(cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config=config)

#Load data
train_images, train_labels = mnist_reader.load_mnist('data/fashion', kind='train')
test_images,test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')

#data preprocessing
num_classes = 10
x_train = train_images.reshape([-1,28,28,1])
x_test= test_images.reshape([-1,28,28,1])[:1000]
y_train = one_hot(train_labels)
y_test = one_hot(test_labels)[:1000]

#parameters
# Network parameters
n_input = 784 # Fashion MNIST data input (img shape: 28*28)
n_classes = 10 # Fashion MNIST total classes (0-9 digits)
image_size = 28
channel_size = 1
n_samples =  x_train.shape[0]
batch_size = 128
epochs = 2
num_iterations = n_samples//batch_size
test_step = 100
learning_rate = 0.01

LOG_DIR = 'sync_logs'
print('parameters specification finished!')

#define network 
def net(x):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        SyncReplicasOptimizer(o
        net = slim.layers.conv2d(x, 32, [3, 3], scope='conv1')
        net = slim.layers.conv2d(x, 64, [3, 3], scope='conv2')
        net = slim.layers.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.layers.conv2d(net, 128, [3, 3], scope='conv3')    
        net = slim.layers.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.layers.flatten(net, scope='flatten')
        net = slim.layers.fully_connected(net, 1024, scope='fully_connected1')
        net = slim.layers.fully_connected(net, 512, scope='fully_connected2')
        net = slim.dropout(net, 0.8, scope = 'dropout1') 
        net = slim.layers.fully_connected(net, 10, activation_fn=None,
                                          scope='pred')
    return net

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    print('Training begin!')
    # Between-graph replication
    is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
    with tf.device(tf.train.replica_device_setter(ps_tasks=1,
        worker_device="/job:worker/task:%d" % FLAGS.task_index)):
        
        # count the number of updates
        global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')

        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None,image_size, image_size,channel_size])
            y = tf.placeholder(tf.float32, [None, n_classes])
            
        Y = net(x)
        with tf.name_scope('train'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = Y, labels = y)
            cost = tf.reduce_mean(cross_entropy)
            
            optimizer = tf.train.AdamOptimizer(learning_rate)
            optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                       replicas_to_aggregate=3,
                                                       total_num_replicas=3)
            opt = optimizer.minimize(cost, global_step=global_step)
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        print('Summaries begin!')
        tf.summary.image('input',x,10)
        tf.summary.scalar('loss',cost) 
        tf.summary.scalar('accuracy',accuracy)
        tf.summary.histogram('pred_y',Y)
        tf.summary.histogram('gradients',opt)
        
        merged = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        
    sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
    stop_hook = tf.train.StopAtStepHook(last_step = 10000)
    summary_hook = tf.train.SummarySaverHook(save_secs=600,output_dir= LOG_DIR,summary_op=merged)
    hooks = [sync_replicas_hook, stop_hook,summary_hook]
    scaff = tf.train.Scaffold(init_op = init_op)
    
    begin_time = time.time()
    print("Waiting for other servers")
    with tf.train.MonitoredTrainingSession(master = server.target,
                                          is_chief = (FLAGS.task_index ==0),
                                          checkpoint_dir = LOG_DIR,
                                          scaffold = scaff,
                                          hooks = hooks) as sess: 
       # tf.reset_default_graph()
        count = 0
        print('Starting training on worker %d'%FLAGS.task_index)

        while not sess.should_stop() and count < 1000:
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph = tf.get_default_graph())
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'test'),graph = tf.get_default_graph())
            
            frequency = 100
            #performe training cycles
            start_time = time.time()

            for count in range(num_iterations*epochs):   
                offset = (count*batch_size)% n_samples
                batch_x = x_train[(offset):(offset+batch_size),:]
                batch_y = y_train[offset:(offset+batch_size),:]
                summary, _,loss,step,acc = sess.run([merged, opt,cost,global_step,accuracy],feed_dict={x: batch_x, y: batch_y})
                train_writer.add_summary(summary,count)
                count += 1
                if count % frequency == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('Worker %d, ' % int(FLAGS.task_index),
                         'At step %d, '% int(step),
                         'Cost: %.4f'% float(loss),
                         'Accuracy: %.4f'% float(acc),
                         'AvgTime: %3.2fms'%float(elapsed_time*100/frequency))
            for i in range(test_step):
                if i % 10 == 0:
                    summary, test_accuracy = sess.run([merged,accuracy],feed_dict={x: x_test, y: y_test})
                    test_writer.add_summary(summary, i)
                    print('Test accuracy at step %s: %s' % (i, test_accuracy))
       

        
    print('done')


