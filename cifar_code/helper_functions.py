from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
#import cPickle
import pickle as cPickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import scipy
import logging
logging.getLogger('tensorflow').setLevel(logging.WARNING)
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from sklearn.utils import shuffle
import datetime

def prep_accuracy_log(continued=False, checkpoint_file_to_use=None, naming=None, variables=None):
    if continued:
        pickle_file_to_continue_from = re.match('(.+?)/', checkpoint_file_to_use).group(1).replace('ckpt_', 'pickle_')
        accuracy_log = cPickle.load(open(pickle_file_to_continue_from, 'rb'))
        naming = accuracy_log['pickle_file'].replace('pickle_', '') + '_contd'
        
        accuracy_log['last_pickle_file'] = accuracy_log['pickle_file']
        accuracy_log['last_tb_log_directory'] = accuracy_log['tb_log_directory']
        accuracy_log['last_checkpoint_directory'] = accuracy_log['checkpoint_directory']

        accuracy_log['checkpoint_file_to_use'] = checkpoint_file_to_use
        accuracy_log['start_step'] = int(re.search('[0-9]+$', checkpoint_file_to_use).group(0))
        index_start_step = accuracy_log['steps'].index(accuracy_log['start_step'])
        max_ave_validation_accuracy = max(accuracy_log['ave_validation'][: index_start_step+1])
        index_max_ave_validation_accuracy = accuracy_log['ave_validation'].index(max_ave_validation_accuracy)
        step_max_ave_validation_accuracy = accuracy_log['steps'][index_max_ave_validation_accuracy]
        accuracy_log['max_ave_validation_accuracy'] = {'step': step_max_ave_validation_accuracy, 
                                                       'accuracy': max_ave_validation_accuracy, 
                                                       'patient_till': max_ave_validation_accuracy - accuracy_log['variables']['early_stopping_patience']}
    else:
        accuracy_log = {}
        accuracy_log['variables'] = variables
            
    accuracy_log['pickle_file'] = 'pickle_' + naming
    accuracy_log['tb_log_directory'] = 'tb_' + naming
    accuracy_log['checkpoint_directory'] = 'ckpt_' + naming  

    #Create a pickle file
    cPickle.dump(accuracy_log, open(accuracy_log['pickle_file'], 'wb'))

    #Create TensorBoard and checkpoint directories
    for d in (accuracy_log['tb_log_directory'], accuracy_log['checkpoint_directory']):
        if tf.gfile.Exists(d):
            tf.gfile.DeleteRecursively(d)
        tf.gfile.MakeDirs(d)

    tf.gfile.MkDir(accuracy_log['checkpoint_directory'] + '/best')
    tf.gfile.MkDir(accuracy_log['checkpoint_directory'] + '/hourly')
    
    return accuracy_log

def get_data_and_labels(accuracy_log):
    enlarge_images = accuracy_log['variables']['enlarge_images']
    crop_images = accuracy_log['variables']['crop_images']
    
    #1 Load image data and labels from the downloaded pickled files

    #1.1 Train data and labels
    train_picles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
    for i in range(len(train_picles)):
        f = open('cifar-10-batches-py/' + train_picles[i], 'rb')
        data_dict = cPickle.load(f,encoding='bytes')
        f.close()
        if i == 0:
            train_data = data_dict[b'data']
            train_labels = data_dict[b'labels']
        else:
            train_data = np.append(train_data, data_dict[b'data'], axis=0)
            train_labels = np.append(train_labels, data_dict[b'labels'])

    #1.2 Validation data and labels
    f = open('cifar-10-batches-py/data_batch_5', 'rb')
    data_dict = cPickle.load(f,encoding='bytes')
    f.close()
    validation_data = data_dict[b'data']
    validation_labels = np.array(data_dict[b'labels'])

    #1.3 Test data and labels
    f = open('cifar-10-batches-py/test_batch', 'rb')
    data_dict = cPickle.load(f,encoding='bytes')
    f.close()
    test_data = data_dict[b'data']
    test_labels = np.array(data_dict[b'labels'])

    #2 One-hot encode the labels
    enc = OneHotEncoder(dtype = np.float32)
    enc.fit(test_labels.reshape(-1,1))
    train_labels = enc.transform(train_labels.reshape(-1,1)).toarray()
    validation_labels = enc.transform(validation_labels.reshape(-1,1)).toarray()
    test_labels = enc.transform(test_labels.reshape(-1,1)).toarray()

    #3 Reshape data into RGB channels
    def into_3channels(data):
        data = np.reshape(data, [-1,3,32,32])
        data = np.transpose(data, [0,2,3,1])
        return data
    train_data = into_3channels(train_data)
    validation_data = into_3channels(validation_data)
    test_data = into_3channels(test_data)

    #4 Enlarge images
    if enlarge_images:
        def enlarge(image):
            return scipy.misc.imresize(image, (64, 64, 3), interp='bicubic')
        train_data = np.array([enlarge(image) for image in train_data])
        validation_data = np.array([enlarge(image) for image in validation_data])
        test_data = np.array([enlarge(image) for image in test_data])

    #5 Convert all values from unit8 to float32
    train_data = train_data.astype(np.float32)/255.0
    validation_data = validation_data.astype(np.float32)/255.0
    test_data = test_data.astype(np.float32)/255.0
    train_labels = train_labels.astype(np.float32)
    validation_labels = validation_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    
    #6 Update accuracy_log
    if enlarge_images:
        accuracy_log['variables']['input_image_side_size'] = 64
    else:
        accuracy_log['variables']['input_image_side_size'] = 32
    if crop_images:
        accuracy_log['variables']['crop_to_side_size'] = int(accuracy_log['variables']['input_image_side_size'] * 3 / 4)
    else:
        accuracy_log['variables']['crop_to_side_size'] = None
    accuracy_log['variables']['train_data_size'] = len(train_data)
    accuracy_log['variables']['steps_per_epoch'] = int(len(train_data) / accuracy_log['variables']['batch_size'])
    accuracy_log['variables']['log_every'] = int(accuracy_log['variables']['steps_per_epoch'] / 2)
    accuracy_log['variables']['print_every'] = accuracy_log['variables']['log_every'] * 10
    cPickle.dump(accuracy_log, open(accuracy_log['pickle_file'], 'wb'))

    return (train_data, validation_data, test_data, train_labels, validation_labels, test_labels)

def change_early_stopping_patience(continued, accuracy_log, to_value):
    accuracy_log['variables']['early_stopping_patience'] = to_value
    if continued:
        accuracy_log['max_ave_validation_accuracy']['patient_till'] = accuracy_log['max_ave_validation_accuracy']['accuracy'] - to_value
    cPickle.dump(accuracy_log, open(accuracy_log['pickle_file'], 'wb'))

def prep_random_modify(image, side):
    image = tf.random_crop(image, [side, side, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63/255.0)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    return image

def random_modify(input_tensor, side):
    return tf.map_fn(lambda x: prep_random_modify(x, side), input_tensor)    

def prep_random_modify_no_crop(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63/255.0)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    return image

def random_modify_no_crop(input_tensor):
    return tf.map_fn(prep_random_modify_no_crop, input_tensor) 

def prep_crop_center(image, side):
    image = tf.image.resize_image_with_crop_or_pad(image, side, side)
    return image

def crop_center(input_tensor, side):
    return tf.map_fn(lambda x: prep_crop_center(x, side), input_tensor)

def prep_random_crop(image, side):
    image = tf.random_crop(image, [side, side, 3])
    return image

def random_crop(input_tensor, side):
    return tf.map_fn(lambda x: prep_random_crop(x, side), input_tensor)    

def batch_normalize(input_tensor, is_training, global_step, scope):
    train_first = tf.logical_and(is_training, tf.equal(global_step, 0))
    return tf.cond(train_first, 
                   lambda: batch_norm(input_tensor, is_training=is_training, center=False, 
                                      updates_collections=None, scope=scope, reuse=None), 
                   lambda: batch_norm(input_tensor, is_training=is_training, center=False, 
                                      updates_collections=None, scope=scope, reuse=True))

def weight_variable(shape, weights_stddev, name):
    initial = tf.truncated_normal(shape, stddev=weights_stddev)
    return tf.Variable(initial, name=name)

def bias_variable(biases_initial, shape, name):
    initial = tf.constant(biases_initial, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def resize_stride2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME')    

def max_pool_3x3_stride2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3_stride1(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    
def train(continued, graph_variables, accuracy_log, train_data, train_labels, validation_data, validation_labels):
    with tf.Session(graph=graph_variables['graph']) as sess:   
    #If assigning devices
    #with tf.Session(graph=graph_variables['graph'], config=tf.ConfigProto(log_device_placement=True, 
    #                                                                      allow_soft_placement=True)) as sess:
        offset = 0
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter(accuracy_log['tb_log_directory'], sess.graph)
        saver_best = tf.train.Saver()
        saver_hourly = tf.train.Saver(max_to_keep=None)  
        last_hourly_save = datetime.datetime.now()
        
        if continued:
            start_step = accuracy_log['start_step']
            saver_hourly.restore(sess, accuracy_log['checkpoint_file_to_use'])
            accuracy_log['index'] = [0]
            accuracy_log['training'] = [accuracy_log['training'][accuracy_log['steps'].index(start_step)]]
            
            correct_validation = 0
            for j in range(int(len(validation_data) / accuracy_log['variables']['batch_size'])):
                validation_batch_data = validation_data[j*accuracy_log['variables']['batch_size']: (j+1)*accuracy_log['variables']['batch_size']]
                validation_batch_labels = validation_labels[j*accuracy_log['variables']['batch_size']: (j+1)*accuracy_log['variables']['batch_size']]
                correct_validation += np.sum(graph_variables['correct_prediction'].eval(feed_dict={graph_variables['data']: validation_batch_data, graph_variables['labels']: validation_batch_labels, graph_variables['keep_prob']: 1.0, graph_variables['is_training']: False}))
            validation_accuracy = correct_validation / len(validation_data)
            
            accuracy_log['validation'] = [validation_accuracy]
            accuracy_log['ave_validation'] = [validation_accuracy]
            accuracy_log['end_time'] = [datetime.datetime.now()]
            accuracy_log['steps'] = [start_step]
            
            start_step += 1
        else:
            start_step = 0
            accuracy_log['index'] = []
            accuracy_log['steps'] = []
            accuracy_log['training'] = []
            accuracy_log['validation'] = []
            accuracy_log['ave_validation'] = []
            accuracy_log['end_time'] = []
            accuracy_log['max_ave_validation_accuracy'] = {'step': -1, 'accuracy': -1, 'patient_till': -1}
            
        for i in range(start_step, accuracy_log['variables']['max_steps']):
            if i == start_step:
                accuracy_log['train_start'] = datetime.datetime.now()
                print('%-10s%s%s' % ('TRAINING ', ' START @ ', accuracy_log['train_start']))
                
            offset = offset % accuracy_log['variables']['train_data_size']
            #Shuffle every epoch
            if offset == 0:
                train_data, train_labels = shuffle(train_data, train_labels)

            if offset <= (accuracy_log['variables']['train_data_size'] - accuracy_log['variables']['batch_size']):
                batch_data = train_data[offset: offset+accuracy_log['variables']['batch_size'], :, :, :]
                batch_labels = train_labels[offset: offset+accuracy_log['variables']['batch_size']]
                offset += accuracy_log['variables']['batch_size']
            else:
                batch_data = train_data[offset: accuracy_log['variables']['train_data_size'], :, :, :]
                batch_labels = train_labels[offset: accuracy_log['variables']['train_data_size']]
                offset = 0
            _, summary = sess.run([graph_variables['optimizer'], graph_variables['summarizer']], 
                                  feed_dict={graph_variables['data']: batch_data, graph_variables['labels']: batch_labels,
                                             graph_variables['keep_prob']: accuracy_log['variables']['dropout_train_keep_prob'], graph_variables['is_training']: True})

            if i % accuracy_log['variables']['log_every'] == 0:
                writer.add_summary(summary, i)
                training_accuracy = graph_variables['accuracy'].eval(feed_dict={graph_variables['data']: batch_data, graph_variables['labels']: batch_labels, 
                                                             graph_variables['keep_prob']: 1.0, graph_variables['is_training']: False})

                correct_validation = 0
                for j in range(int(len(validation_data) / accuracy_log['variables']['batch_size'])):
                    validation_batch_data = validation_data[j*accuracy_log['variables']['batch_size']: (j+1)*accuracy_log['variables']['batch_size']]
                    validation_batch_labels = validation_labels[j*accuracy_log['variables']['batch_size']: (j+1)*accuracy_log['variables']['batch_size']]
                    correct_validation += np.sum(graph_variables['correct_prediction'].eval(feed_dict={graph_variables['data']: validation_batch_data, graph_variables['labels']: validation_batch_labels, graph_variables['keep_prob']: 1.0, graph_variables['is_training']: False}))
                validation_accuracy = correct_validation / len(validation_data)
                
                accuracy_log['index'].append(len(accuracy_log['index']))
                accuracy_log['steps'].append(i)
                accuracy_log['training'].append(training_accuracy)
                accuracy_log['validation'].append(validation_accuracy)
                if len(accuracy_log['validation']) < accuracy_log['variables']['average_n_validation_accuracy']:
                    start = 0
                    num = len(accuracy_log['validation'])
                else:
                    start = len(accuracy_log['validation']) - accuracy_log['variables']['average_n_validation_accuracy']
                    num = accuracy_log['variables']['average_n_validation_accuracy']
                n_recent_validation_accuracy_log = accuracy_log['validation'][start: start+num]
                ave_validation_accuracy = sum(n_recent_validation_accuracy_log)/float(num)
                accuracy_log['ave_validation'].append(ave_validation_accuracy)
                accuracy_log['end_time'].append(datetime.datetime.now())

                #Save a model and a pickle file every hour
                if last_hourly_save + datetime.timedelta(hours=1) < datetime.datetime.now():                    
                    path_checkpoint_file = saver_hourly.save(sess, accuracy_log['checkpoint_directory'] + '/hourly/model', global_step=i, latest_filename='hourly_checkpoint')

                    cPickle.dump(accuracy_log, open(accuracy_log['pickle_file'], 'wb'))
                    print('<< hourly save at %s and %s >>' % (path_checkpoint_file, accuracy_log['pickle_file']))
                    last_hourly_save = datetime.datetime.now()
                
                #If it is the best model so far
                if ave_validation_accuracy >= accuracy_log['max_ave_validation_accuracy']['accuracy']:
                    accuracy_log['max_ave_validation_accuracy']['step'] = i
                    accuracy_log['max_ave_validation_accuracy']['accuracy'] = ave_validation_accuracy
                    accuracy_log['max_ave_validation_accuracy']['patient_till'] = ave_validation_accuracy - accuracy_log['variables']['early_stopping_patience']

                    path_checkpoint_file = saver_best.save(sess, accuracy_log['checkpoint_directory'] + '/best/model', global_step=i, latest_filename='best_checkpoint')
                    print('<< best model so far is saved in %s >> ave validation accuracy %7.5f' % 
                          (path_checkpoint_file, ave_validation_accuracy))
                    cPickle.dump(accuracy_log, open(accuracy_log['pickle_file'], 'wb'))

                #If not, and if patience is over
                elif (i > accuracy_log['variables']['start_step_early_stopping']) & (ave_validation_accuracy < accuracy_log['max_ave_validation_accuracy']['patient_till']):
                    print('** reached early stopping patience **')
                    break

            if i % accuracy_log['variables']['print_every'] == 0:
                print('STEP %7d%s%s'%(i, ' END @ ', datetime.datetime.now()) +
                      ', training accuracy %7.5f, ave validation accuracy %7.5f' % (training_accuracy, ave_validation_accuracy))

        accuracy_log['train_end'] = datetime.datetime.now()
        print('%-12s%s%s' % ('TRAINING ', ' END @ ', accuracy_log['train_end']))

def test_accuracy(graph_variables, accuracy_log, test_data, test_labels):
    with tf.Session(graph=graph_variables['graph']) as sess:
        tf.global_variables_initializer().run()
        #Grab the model with the highest ave validation accuracy to evaluate test accuracy
        accuracy_log['best_model'] = tf.train.latest_checkpoint(accuracy_log['checkpoint_directory'] + '/best', 
                                                                latest_filename='best_checkpoint')
        saver = tf.train.Saver()
        saver.restore(sess, accuracy_log['best_model'])
        
        correct_test = 0
        for j in range(int(len(test_data) / accuracy_log['variables']['batch_size'])):
            test_batch_data = test_data[j*accuracy_log['variables']['batch_size']: (j+1)*accuracy_log['variables']['batch_size']]
            test_batch_labels = test_labels[j*accuracy_log['variables']['batch_size']: (j+1)*accuracy_log['variables']['batch_size']]
            correct_test += np.sum(graph_variables['correct_prediction'].eval(feed_dict={graph_variables['data']: test_batch_data, graph_variables['labels']: test_batch_labels, graph_variables['keep_prob']: 1.0, graph_variables['is_training']: False}))
        test_accuracy = correct_test / len(test_data)
        
        accuracy_log['test_accuracy'] = test_accuracy
        print('<< test accuracy %7.5f >>' % accuracy_log['test_accuracy'])
        cPickle.dump(accuracy_log, open(accuracy_log['pickle_file'], 'wb'))
