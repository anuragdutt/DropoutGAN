
# coding: utf-8

# In[1]:


import pickle   
import numpy as np  
import os  
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math
from tensorflow.contrib.layers import batch_norm


# In[2]:


class Cifar100DataReader():  
    def __init__(self,cifar_folder,onehot=True):  
        self.cifar_folder=cifar_folder  
        self.onehot=onehot  
        self.data_label_train=None            
        self.data_label_test=None             
        self.batch_index=0                    
        self.test_batch_index=0                
        f=os.path.join(self.cifar_folder,"train")  
        print ('read: %s'%f  )
        fo = open(f, 'rb')
        self.dic_train = pickle.load(fo,encoding='bytes')
        fo.close()
        self.data_label_train=list(zip(self.dic_train[b'data'],self.dic_train[b'fine_labels']) ) #label 0~99  
        np.random.shuffle(self.data_label_train)           
 
    
    def dataInfo(self):
        print (self.data_label_train[0:2] )
        print (self.dic_train.keys())
        print (b"coarse_labels:",len(self.dic_train[b"coarse_labels"]))
        print (b"filenames:",len(self.dic_train[b"filenames"]))
        print (b"batch_label:",len(self.dic_train[b"batch_label"]))
        print (b"fine_labels:",len(self.dic_train[b"fine_labels"]))
        print (b"data_shape:",np.shape((self.dic_train[b"data"])))
        print (b"data0:",type(self.dic_train[b"data"][0]))
 
 
   
    def next_train_data(self,batch_size=100):  
        """ 
        return list of numpy arrays [na,...,na] with specific batch_size 
                na: N dimensional numpy array  
        """            
        if self.batch_index<len(self.data_label_train)/batch_size:  
            #print ("batch_index:",self.batch_index  )
            datum=self.data_label_train[self.batch_index*batch_size:(self.batch_index+1)*batch_size]  
            self.batch_index+=1  
            return self._decode(datum,self.onehot)  
        else:  
            self.batch_index=0  
            np.random.shuffle(self.data_label_train)  
            datum=self.data_label_train[self.batch_index*batch_size:(self.batch_index+1)*batch_size]  
            self.batch_index+=1  
            return self._decode(datum,self.onehot) 
    
    def _decode(self,datum,onehot):  
        rdata=list()     
        rlabel=list()  
        if onehot:  
            for d,l in datum:  
                rdata.append(np.reshape(np.reshape(d,[3,1024]).T,[32,32,3]))   
                hot=np.zeros(100)    
                hot[int(l)]=1            
                rlabel.append(hot)  
        else:  
            for d,l in datum:  
                rdata.append(np.reshape(np.reshape(d,[3,1024]).T,[32,32,3]))  
                rlabel.append(int(l))  
        return rdata,rlabel  
           
    def next_test_data(self,batch_size=100):  
        ''''' 
        return list of numpy arrays [na,...,na] with specific batch_size 
                na: N dimensional numpy array  
        '''  
        if self.data_label_test is None:  
            f=os.path.join(self.cifar_folder,"test")  
            print ('read: %s'%f  )
            fo = open(f, 'rb')
            dic_test = pickle.load(fo,encoding='bytes')
            fo.close()           
            data=dic_test[b'data']            
            labels=dic_test[b'fine_labels']   # 0 ~ 99  
            self.data_label_test=list(zip(data,labels) )
            self.batch_index=0
 
        if self.test_batch_index<len(self.data_label_test)/batch_size:  
            #print ("test_batch_index:",self.test_batch_index )
            datum=self.data_label_test[self.test_batch_index*batch_size:(self.test_batch_index+1)*batch_size]  
            self.test_batch_index+=1  
            return self._decode(datum,self.onehot)  
        else:  
            self.test_batch_index=0  
            np.random.shuffle(self.data_label_test)  
            datum=self.data_label_test[self.test_batch_index*batch_size:(self.test_batch_index+1)*batch_size]  
            self.test_batch_index+=1  
            return self._decode(datum,self.onehot)    


# In[3]:



def CNN():
    sess=tf.InteractiveSession()
    
    max_steps=3000    
    batch_size=100  
 
   
    image_holder=tf.placeholder(tf.float32,[batch_size,32,32,3]) 
    label_holder=tf.placeholder(tf.float32,[batch_size,100])
    keep_prob = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)
    is_training = tf.placeholder(tf.bool)
    phase_train = tf.placeholder(tf.bool)
 
    
    be=0.2
    #ConvNet1
    weight1=variable_with_weight_loss(shape=[5,5,3,96],stddev=5e-2,w1=0.0) 
    kernel1=tf.nn.conv2d(image_holder,weight1,[1,1,1,1],padding='SAME') 
    conv1_bn = batch_normalize(kernel1, is_training=is_training, global_step=global_step,scope="con1_bn")
    r_conv1 = tf.random_uniform([batch_size],minval=1-be,maxval=1+be,dtype=tf.float32)
    conv1_drop = tf.cond(phase_train,lambda:tf.multiply(conv1_bn,r_conv1[:,tf.newaxis,tf.newaxis,tf.newaxis]),lambda:conv1_bn)
    #conv1_drop=tf.nn.dropout(conv1_bn, keep_prob)
    bias1=tf.Variable(tf.constant(0.0,shape=[96]))   
    conv1=tf.nn.relu(tf.nn.bias_add(conv1_drop,bias1))
    pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME') 
    norm1=tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75) 
 
    #ConvNet2
    weight2=variable_with_weight_loss(shape=[5,5,96,128],stddev=5e-2,w1=0.0) 
    kernel2=tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding='SAME') 
    conv2_bn = batch_normalize(kernel2, is_training=is_training,global_step=global_step, scope="con2_bn")
    r_conv2 = tf.random_uniform([batch_size],minval=1-be,maxval=1+be,dtype=tf.float32)
    conv2_drop = tf.cond(phase_train,lambda:tf.multiply(conv2_bn,r_conv2[:,tf.newaxis,tf.newaxis,tf.newaxis]),lambda:conv2_bn)
    #conv2_drop=tf.nn.dropout(conv2_bn, keep_prob)
    bias2=tf.Variable(tf.constant(0.1,shape=[128]))  
    conv2=tf.nn.relu(tf.nn.bias_add(conv2_drop,bias2)) 
    norm2=tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
    pool2=tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME') 
    
    #ConvNet3
    weight3=variable_with_weight_loss(shape=[5,5,128,256],stddev=5e-2,w1=0.0) 
    kernel3=tf.nn.conv2d(pool2,weight3,[1,1,1,1],padding='SAME') 
    conv3_bn = batch_normalize(kernel3, is_training=is_training,global_step=global_step, scope="con3_bn")
    r_conv3 = tf.random_uniform([batch_size],minval=1-be,maxval=1+be,dtype=tf.float32)
    conv3_drop = tf.cond(phase_train,lambda:tf.multiply(conv3_bn,r_conv3[:,tf.newaxis,tf.newaxis,tf.newaxis]),lambda:conv3_bn)
    bias3=tf.Variable(tf.constant(0.1,shape=[256]))  
    conv3=tf.nn.relu(tf.nn.bias_add(conv3_bn,bias3)) 
    norm3=tf.nn.lrn(conv3,4,bias=1.0,alpha=0.001/9.0,beta=0.75) 
    pool3=tf.nn.max_pool(norm3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 
    
 
    #Fc1
    reshape=tf.reshape(pool2,[batch_size,-1])  
    dim=reshape.get_shape()[1].value            
    weight4=variable_with_weight_loss(shape=[dim,2048],stddev=0.04,w1=0.004)  
    bias4=tf.Variable(tf.constant(0.1,shape=[2048]))
    fc1_drop = tf.cond(phase_train,lambda:tf.nn.dropout(reshape,keep_prob),lambda:reshape)
    #fc1_drop=tf.nn.dropout(reshape, keep_prob)
    fc1=tf.matmul(fc1_drop,weight4)+bias4
    fc1_relu=tf.nn.relu(fc1)
    #local3=tf.nn.relu(tf.matmul(reshape,weight3)+bias3)
    
    
    #Fc2
    weight5=variable_with_weight_loss(shape=[2048,2048],stddev=0.04,w1=0.004) 
    bias5=tf.Variable(tf.constant(0.1,shape=[2048]))
    fc2_drop=tf.cond(phase_train, lambda: tf.nn.dropout(fc1_relu, keep_prob), lambda:fc1_relu)
    fc2=tf.matmul(fc2_drop,weight5)+bias5
    fc2_relu=tf.nn.relu(fc2)
    #local4=tf.nn.relu(tf.matmul(local3,weight4)+bias4)
    

    
    #softmax
    weight6=variable_with_weight_loss(shape=[2048,100],stddev=1/2048.0,w1=0.0) 
    bias6=tf.Variable(tf.constant(0.0,shape=[100]))
    logits=tf.nn.softmax(tf.matmul(fc2_relu,weight6)+bias6)
 
        
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(label_holder*tf.log(logits),reduction_indices=[1]))
    tf.add_to_collection('losses',cross_entropy)  
    loss=tf.add_n(tf.get_collection('losses'))    
 
    train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)  
    # top_k_op=tf.nn.in_top_k(logits,label_holder,1)  
 
    # Train the model
    sess=tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print ("Training begin......")
    cifar100=Cifar100DataReader(cifar_folder="cifar-100-python")
    for step in range(max_steps):
        start=time.time()
        image_batch,label_batch=cifar100.next_train_data(batch_size=batch_size)
        train_op.run(feed_dict={image_holder:image_batch,label_holder:label_batch,keep_prob:0.8,is_training:True,
                                global_step:0,phase_train:True})
    
    print ("training end.")
    print ("caculate precision......")
    
 
    # Test set
    num_example=10000     
    num_iter=int(math.ceil(num_example/batch_size))  
    true_count=0
    total_sample_count=num_iter*batch_size  
    step=0
    while step<num_iter:
        test_data,test_label=cifar100.next_test_data(batch_size=batch_size) 
        correction_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(label_holder,1))     
        correction=sess.run([correction_prediction],feed_dict={image_holder:test_data,label_holder:test_label, keep_prob:1, 
                                                               is_training:False,global_step:1,phase_train:False})
        true_count+=np.sum(correction)
        step+=1
    precision=true_count/total_sample_count
    print ("precision:",precision)
  
    
    saver = tf.train.Saver() 
    save_path = saver.save(sess,"./Cifar100/model.ckpt")  
    print("save model:{0} Finished".format(save_path))      


# In[4]:



def loss(logits,labels):
    labels=tf.cast(labels,tf.int64)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')
    cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_colletion('losses'),name='tatol_loss')

def batch_normalize(input_tensor, is_training, global_step, scope):
    train_first = tf.logical_and(is_training, tf.equal(global_step, 0))
    return tf.cond(train_first, 
                   lambda: batch_norm(input_tensor, is_training=is_training, center=False, 
                                      updates_collections=None, scope=scope, reuse=None), 
                   lambda: batch_norm(input_tensor, is_training=is_training, center=False, 
                                      updates_collections=None, scope=scope, reuse=True)) 
 


def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss') 
        tf.add_to_collection('losses',weight_loss)  
    return var
 
if __name__=='__main__':
    CNN()

