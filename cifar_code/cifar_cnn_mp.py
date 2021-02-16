
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from helper_functions import *


# In[2]:



continued = False
checkpoint_file_to_use = 'ckpt_model_01_run01/hourly/model-9350'

#If training from step 0
#############################
naming = 'model_02_run01'
#############################
variables = {
    'modify_images': False,
    'crop_images': False,
    'enlarge_images': False,
    'before_flatten_image_side_size': 4, #Need to calculate
    'first_hidden_layer_features': 384,
    'weights_stddev': 0.015,
    'biases_initial': 0.1,
    'dropout_train_keep_prob': 1.0,
    'learning_rate_initial': 0.01,
    'learning_rate_decay_steps': int(10000000),
    'learning_rate_decay': 1.0,
    'start_step_early_stopping': 150000,
    'early_stopping_patience': 0.1,
    'batch_size': 200,
    'max_steps': 100000000,
    'average_n_validation_accuracy': 8
}


# In[3]:


#Create accuracy_log to pickle and directories for TensorBoard and checkpoint
accuracy_log = prep_accuracy_log(continued, checkpoint_file_to_use, naming, variables)


# In[4]:


#Get data and labels
(train_data, validation_data, test_data, train_labels, validation_labels, test_labels) = get_data_and_labels(accuracy_log)


# In[5]:


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
        
    W_conv1 = weight_variable([3, 3, 3, 64], weights_stddev, 'W_conv1')
    b_conv1 = bias_variable(biases_initial, [64], 'b_conv1')
    conv1 = conv2d(data_v2, W_conv1) + b_conv1
    conv1_relu = tf.nn.relu(conv1)
    pool1 = max_pool_3x3_stride2(conv1_relu)

    W_conv2 = weight_variable([3, 3, 64, 64], weights_stddev, 'W_conv2')
    b_conv2 = bias_variable(biases_initial, [64], 'b_conv2')
    conv2 = conv2d(pool1, W_conv2) + b_conv2
    conv2_relu = tf.nn.relu(conv2)
    pool2 = max_pool_3x3_stride2(conv2_relu)

    W_conv3 = weight_variable([3, 3, 64, 128], weights_stddev, 'W_conv3')
    b_conv3 = bias_variable(biases_initial, [128], 'b_conv3')
    conv3 = conv2d(pool2, W_conv3) + b_conv3
    conv3_relu = tf.nn.relu(conv3)
    pool3 = max_pool_3x3_stride2(conv3_relu)
    
    pool3_flat = tf.reshape(pool3, [-1, before_flatten_image_side_size*before_flatten_image_side_size*128])

    W_fc1 = weight_variable([before_flatten_image_side_size*before_flatten_image_side_size*128, first_hidden_layer_features], weights_stddev, 'W_fc1')
    b_fc1 = bias_variable(biases_initial, [first_hidden_layer_features], 'b_fc1')
    fc1 = tf.matmul(pool3_flat, W_fc1) + b_fc1
    fc1_relu = tf.nn.relu(fc1)
    
    W_fc2 = weight_variable([first_hidden_layer_features, int(first_hidden_layer_features/2)], weights_stddev, 'W_fc2')
    b_fc2 = bias_variable(biases_initial, [first_hidden_layer_features/2], 'b_fc2')
    fc2 = tf.matmul(fc1_relu, W_fc2) + b_fc2
    fc2_relu = tf.nn.relu(fc2)
    
    W_fc3 = weight_variable([int(first_hidden_layer_features/2), 10], weights_stddev, 'W_fc3')
    b_fc3 = bias_variable(biases_initial, [10], 'b_fc3')
    logits = tf.matmul(fc2_relu, W_fc3) + b_fc3

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lables,logits=logits))
    
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
                   'correct_prediction': correct_prediction,
                   'accuracy': accuracy}


# In[6]:


#Train the model
train(continued, graph_variables, accuracy_log, train_data, train_labels, validation_data, validation_labels)


# In[7]:


#Test accuracy
test_accuracy(graph_variables, accuracy_log, test_data, test_labels)

