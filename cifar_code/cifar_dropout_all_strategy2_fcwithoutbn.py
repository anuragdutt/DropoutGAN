
# coding: utf-8

# In[16]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from helper_functions1 import *


# In[17]:


continued = False
checkpoint_file_to_use = None 

#If training from step 0
#############################
naming = 'cifar10_dropout_inalllayers'
#############################
variables = {
    'modify_images': False,
    'crop_images': False,
    'enlarge_images': False,
    'before_flatten_image_side_size': 4, #Need to calculate
    'first_hidden_layer_features': 384,
    'weights_stddev': 0.015,
    'biases_initial': 0.1,
    'dropout_train_keep_prob': 0.5,
    'learning_rate_initial': 0.1,
    'learning_rate_decay_steps': int(10000000),
    'learning_rate_decay': 1.0,
    'start_step_early_stopping': 150000,
    'early_stopping_patience': 0.1,
    'batch_size': 400,
    'max_steps': 50000,
    'average_n_validation_accuracy': 8
}


# In[18]:


#Create accuracy_log to pickle and directories for TensorBoard and checkpoint
accuracy_log = prep_accuracy_log(continued, checkpoint_file_to_use, naming, variables)


# In[19]:


#Get data and labels
(train_data, validation_data, test_data, train_labels, validation_labels, test_labels) = get_data_and_labels(accuracy_log)


# In[20]:


#Build a model
crop_images = accuracy_log['variables']['crop_images']
modify_images = accuracy_log['variables']['modify_images']
weights_stddev = accuracy_log['variables']['weights_stddev']
biases_initial = accuracy_log['variables']['biases_initial']
input_image_side_size = accuracy_log['variables']['input_image_side_size']
crop_to_side_size = accuracy_log['variables']['crop_to_side_size']
before_flatten_image_side_size = accuracy_log['variables']['before_flatten_image_side_size']
first_hidden_layer_features = accuracy_log['variables']['first_hidden_layer_features']
learning_rate_initial = accuracy_log['variables']['learning_rate_initial']
learning_rate_decay_steps = accuracy_log['variables']['learning_rate_decay_steps']
learning_rate_decay = accuracy_log['variables']['learning_rate_decay']

graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool)

    data = tf.placeholder(tf.float32, [None, input_image_side_size, input_image_side_size, 3])
    labels = tf.placeholder(tf.float32, [None, 10])
    
    if crop_images & modify_images:
        data_v2 = tf.cond(is_training, lambda: random_modify(data, crop_to_side_size), lambda: crop_center(data, crop_to_side_size))
    elif (not crop_images) & modify_images:
        data_v2 = tf.cond(is_training, lambda: random_modify_no_crop(data), lambda: tf.identity(data))
    elif crop_images & (not modify_images):
        data_v2 = tf.cond(is_training, lambda: random_crop(data, crop_to_side_size), lambda: crop_center(data, crop_to_side_size))
    else:
        data_v2 = tf.identity(data)
 
    
    be=0.1
    # ConNet1
    #data_v2 = batch_normalize(data_v2, is_training=is_training, global_step=global_step, scope='bn_data_v2')
    r_cnn1 = tf.random_uniform([400],minval=1-be,maxval=1+be,dtype=tf.float32)
    conv1_drop = tf.cond(phase_train,lambda:tf.multiply(data_v2,r_cnn1[:,tf.newaxis,tf.newaxis,tf.newaxis]),lambda:data_v2)
  
    
    #conv1_drop = tf.nn.dropout(data_v2, keep_prob)
    W_conv1 = weight_variable([3, 3, 3, 64], weights_stddev, 'W_conv1')
    b_conv1 = bias_variable(biases_initial, [64], 'b_conv1')
    conv1 = conv2d(conv1_drop, W_conv1) + b_conv1
    conv1_relu = tf.nn.relu(conv1)
    pool1 = max_pool_3x3_stride2(conv1_relu)
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


    # ConvNet2
    #norm1 = batch_normalize(norm1, is_training=is_training, global_step=global_step, scope='bn_norm1')
    r_cnn2 = tf.random_uniform([400],minval=1-be,maxval=1+be,dtype=tf.float32)
    conv2_drop = tf.cond(phase_train,lambda:tf.multiply(norm1,r_cnn2[:,tf.newaxis,tf.newaxis,tf.newaxis]),lambda:norm1)
    
    #conv2_drop = tf.nn.dropout(norm1, keep_prob)
    W_conv2 = weight_variable([3, 3, 64, 64], weights_stddev, 'W_conv2')
    b_conv2 = bias_variable(biases_initial, [64], 'b_conv2')
    conv2 = conv2d(conv2_drop, W_conv2) + b_conv2
    conv2_relu = tf.nn.relu(conv2)
    pool2 = max_pool_3x3_stride2(conv2_relu)
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # ConvNet3
    #norm2 = batch_normalize(norm2, is_training=is_training, global_step=global_step, scope='bn_norm2')
    r_cnn3 = tf.random_uniform([400],minval=1-be,maxval=1+be,dtype=tf.float32)
    conv3_drop = tf.cond(phase_train,lambda:tf.multiply(norm2,r_cnn3[:,tf.newaxis,tf.newaxis,tf.newaxis]),lambda:norm2)
    
    #conv3_drop = tf.nn.dropout(norm2, keep_prob)
    W_conv3 = weight_variable([3, 3, 64, 128], weights_stddev, 'W_conv3')
    b_conv3 = bias_variable(biases_initial, [128], 'b_conv3')
    conv3 = conv2d(conv3_drop, W_conv3) + b_conv3
    conv3_relu = tf.nn.relu(conv3)
    pool3 = max_pool_3x3_stride2(conv3_relu)
    norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    
    norm3_flat = tf.reshape(norm3, [-1, before_flatten_image_side_size*before_flatten_image_side_size*128])
   

    # FC1
    #norm3_flat = batch_normalize(norm3_flat, is_training=is_training, global_step=global_step, scope='bn_norm3_flat')
    #norm3_flat_drop = tf.nn.dropout(norm3_flat, keep_prob)
    #r_fc1=tf.random_uniform([400],minval=1-be,maxval=1+be,dtype=tf.float32)
    norm3_flat_drop = tf.cond(phase_train,lambda:tf.nn.dropout(norm3_flat,keep_prob),lambda:norm3_flat)
    #norm3_flat_drop=tf.cond(phase_train,lambda:tf.multiply(norm3_flat,r_fc1[:,tf.newaxis,tf.newaxis,tf.newaxis]),lambda:norm3_flat)
    W_fc1 = weight_variable([before_flatten_image_side_size*before_flatten_image_side_size*128, first_hidden_layer_features], weights_stddev, 'W_fc1')
    b_fc1 = bias_variable(biases_initial, [first_hidden_layer_features], 'b_fc1')
    fc1 = tf.matmul(norm3_flat_drop, W_fc1) + b_fc1
    fc1_relu = tf.nn.relu(fc1)

    # FC2
    #fc1_relu = batch_normalize(fc1_relu, is_training=is_training, global_step=global_step, scope='bn_fc1_relu')    
    #fc1_relu_drop = tf.nn.dropout(fc1_relu, keep_prob)
    
    #r_fc2=tf.random_uniform([400],minval=1-be,maxval=1+be,dtype=tf.float32)
    #fc1_relu_drop=tf.cond(phase_train,lambda:tf.multiply(fc1_relu,r_fc2[:,tf.newaxis,tf.newaxis,tf.newaxis]),lambda:fc1_relu)
    
    fc1_relu_drop = tf.cond(phase_train,lambda:tf.nn.dropout(fc1_relu,keep_prob),lambda:fc1_relu)
    W_fc2 = weight_variable([first_hidden_layer_features, int(first_hidden_layer_features/2)], weights_stddev, 'W_fc2')
    b_fc2 = bias_variable(biases_initial, [first_hidden_layer_features/2], 'b_fc2')
    fc2 = tf.matmul(fc1_relu_drop, W_fc2) + b_fc2
    fc2_relu = tf.nn.relu(fc2)
    
    # FC3
    #fc2_relu = batch_normalize(fc2_relu, is_training=is_training, global_step=global_step, scope='bn_fc2_relu')
    #fc2_relu_drop = tf.nn.dropout(fc2_relu, keep_prob)
    
    #r_fc3=tf.random_uniform([400],minval=1-be,maxval=1+be,dtype=tf.float32)
    #fc2_relu_drop=tf.cond(phase_train,lambda:tf.multiply(fc2_relu,r_fc3[:,tf.newaxis,tf.newaxis,tf.newaxis]),lambda:fc2_relu)
    
    fc2_relu_drop = tf.cond(phase_train,lambda:tf.nn.dropout(fc2_relu,keep_prob),lambda:fc2_relu)
    W_fc3 = weight_variable([int(first_hidden_layer_features/2), 10], weights_stddev, 'W_fc3')
    b_fc3 = bias_variable(biases_initial, [10], 'b_fc3')
    logits = tf.matmul(fc2_relu_drop, W_fc3) + b_fc3


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    learning_rate = tf.train.exponential_decay(learning_rate_initial, global_step, learning_rate_decay_steps, 
                                               learning_rate_decay, staircase=True)
    
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('training_accuracy', accuracy)
    
    
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    
    summarizer = tf.summary.merge_all()

    
graph_variables = {'graph': graph, 
                   'optimizer': optimizer, 
                   'summarizer': summarizer, 
                   'data': data, 
                   'labels': labels, 
                   'keep_prob': keep_prob, 
                   'is_training': is_training, 
                   'phase_train': phase_train,
                   'correct_prediction': correct_prediction,
                   'accuracy': accuracy}


# In[21]:


#Train the model
train(continued, graph_variables, accuracy_log, train_data, train_labels, validation_data, validation_labels)


# In[22]:


#Test accuracy
test_accuracy(graph_variables, accuracy_log, test_data, test_labels)


# In[ ]:




