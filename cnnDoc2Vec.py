import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import corpora

from scipy.io import loadmat

import datetime

class cnnDoc2Vec:
    def __init__(self, maxWordNum = 24, wordEmbeddingDim = 300, batchSize = 100, dropoutRate = 0.3):
        self.maxWordNum = maxWordNum
        self.wordEmbeddingDim = wordEmbeddingDim
        self.batchSize = batchSize
        self.dropoutRate = dropoutRate
        self.data_place_holder = tf.placeholder(tf.float32,[self.batchSize,self.wordEmbeddingDim,self.maxWordNum,1])
        # self.label_place_holder = tf.placeholder(tf.float32, [batchSize])
        # self.train_label_one_hot = tf.one_hot(tf.cast(self.label_place_holder, dtype = tf.int32), 2)
        self.inference()
        # logits = self.softmax_linear
        # self.acc_op = self.accuracy(logits, self.train_label_one_hot)
        # self.loss_op = self.loss(logits, self.train_label_one_hot)
    def inference(self):
        with tf.variable_scope('conv1') as scope:
        #var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
            try:
                kernel = tf.get_variable(
                name = "weights", 
                shape = [self.wordEmbeddingDim, 5, 1, 128], 
                initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype=tf.float32),
                dtype = tf.float32)
            except Exception as e:
                print (e)
                scope.reuse_variables()
                kernel = tf.get_variable(name = "weights")
            
            data_place_holder = tf.pad(self.data_place_holder, [[0,0],[0,0],[2,2],[0,0]])

            conv = tf.nn.conv2d(
            input=data_place_holder,
            filter = kernel,
            strides = [1, 1, 1, 1],
            padding='VALID',
            data_format = "NHWC")

            try:
                biases = tf.get_variable(
                    name = "biases",
                    shape= [128],
                    initializer = tf.constant_initializer(0.0, dtype = tf.float32),
                    dtype=tf.float32)
            except Exception as e:
                scope.reuse_variables()
                biases = tf.get_variable(name = "biases")

            conv_with_biases = tf.nn.bias_add(conv, biases)

            #Dropout layer
            pre_activation = tf.nn.dropout(conv_with_biases, keep_prob = 1 - self.dropoutRate)

            self.conv1 = tf.nn.relu(pre_activation, name=scope.name)

            #_activation_summary(conv1)
        

            self.pool1 = tf.nn.max_pool(self.conv1, ksize = [1, 1, 2, 1], strides = [1, 1, 2, 1],
                                padding = "VALID", name = 'pool1')  
        
        #Conv Block 2
        with tf.variable_scope('conv2') as scope:
            try:
                kernel = tf.get_variable(
                name = 'weights',
                shape = [1, 5, 128, 256],
                initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
                dtype = tf.float32)
            except Exception as e:
                scope.reuse_variables()
                kernel = tf.get_variable(name = "weights")

            pool1 = tf.pad(self.pool1, [[0,0], [0,0], [2,2], [0,0]])

            conv = tf.nn.conv2d(
            input = pool1,
            filter = kernel,
            strides = [1, 1, 1, 1],
            padding = 'VALID',
            data_format = 'NHWC')
            
            try:
                biases = tf.get_variable(
                name = 'biases',
                shape = [256],
                initializer = tf.constant_initializer(0.0, dtype = tf.float32),
                dtype = tf.float32)
            except Exception as e:
                scope.reuse_variables()
                biases = tf.get_variable(name = "biases")

            conv_with_biases = tf.nn.bias_add(conv, biases)

            #Dropout
            pre_activation = tf.nn.dropout(conv_with_biases, keep_prob = 1 - self.dropoutRate)

            self.conv2 = tf.nn.relu(pre_activation, name = scope.name)
            
            #_activation_summary(conv2)

            self.pool2 = tf.nn.max_pool(self.conv2, ksize = [1, 1, 2, 1], strides = [1, 1, 2, 1],
                                padding = "VALID", name = 'pool2')
        
        #Conv Block 3
        with tf.variable_scope('conv3') as scope:
            try:
                kernel = tf.get_variable(
                name = 'weights',
                shape = [1, 5, 256, 512],
                initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
                dtype = tf.float32)
            except Exception as e:
                scope.reuse_variables()
                kernel = tf.get_variable(name = "weights")

            pool2 = tf.pad(self.pool2, [[0,0], [0,0], [2,2], [0,0]])

            conv = tf.nn.conv2d(
            input = pool2,
            filter = kernel,
            strides = [1, 1, 1, 1],
            padding = 'VALID',
            data_format = 'NHWC')

            #use_bias = false
            
            pre_activation = tf.nn.dropout(conv, keep_prob = 1 - self.dropoutRate)

            self.conv3 = tf.nn.relu(pre_activation, name = scope.name)

            #_activation_summary(conv3)

            self.pool3 = tf.nn.max_pool(self.conv3, ksize = [1, 1, 2, 1], strides = [1, 1, 2, 1],
                                padding = "VALID", name = 'pool3')

        
        #Conv Block 4
        with tf.variable_scope('conv4') as scope:
            try:
                kernel = tf.get_variable(
                name = 'weights',
                shape = [1, 3, 512, 1],
                initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
                dtype = tf.float32)
            except Exception as e:
                scope.reuse_variables()
                kernel = tf.get_variable(name = "weights")

            conv = tf.nn.conv2d(
            input=self.pool3,
            filter = kernel,
            strides = [1, 1, 1, 1],
            padding='VALID',
            data_format = "NHWC")
            try:
                biases = tf.get_variable(
                    name = "biases",
                    shape= [1],
                    initializer = tf.constant_initializer(0.0, dtype = tf.float32),
                    dtype=tf.float32)
            except Exception as e:
                scope.reuse_variables()
                biases = tf.get_variable(name = "biases")

            self.res = tf.nn.bias_add(conv, biases)    

        #Add a Softmax Layer
        with tf.variable_scope('softmax_layer') as scope:
            try:
                W = tf.get_variable(name = 'weights', shape = [1, 2],
                initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
                dtype = tf.float32)
            except Exception as e:    
                scope.reuse_variables()
                W = tf.get_variable(name = "weights")
            try:
                b = tf.get_variable(name = 'biases', shape = [2], 
                initializer = tf.constant_initializer(0.0),
                dtype = tf.float32)
            except Exception as e:
                scope.reuse_variables()
                b = tf.get_variable(name = "biases")
            res_reshape = tf.reshape(self.res, [self.batchSize, 1])
            self.softmax_linear = tf.add(tf.matmul(res_reshape, W), b, name=scope.name)


    def loss(self, logits, labels):
        #losses = tf.nn.sigmoid_cross_entropy_with_logits(logits = tf.reshape(logits, [100]), labels = labels)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
        return tf.reduce_mean(losses)

    def accuracy(self, logits, labels):
        predictions = tf.argmax(logits, 1, name="predictions")
        correct_predictions = tf.equal(predictions, tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    # sess = tf.Session()
    
    # data = np.zeros([100,300,24,1])
    # cnn = cnnDoc2vec()
    # sess.run(tf.initialize_all_variables())
    # print sess.run(cnn.conv1, feed_dict={cnn.data_place_holder: data}).shape