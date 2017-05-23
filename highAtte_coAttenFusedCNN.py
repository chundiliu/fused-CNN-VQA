import tensorflow as tf
import numpy as np
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import corpora
from scipy.io import loadmat
import datetime
import pandas as pd
import os, h5py, sys, argparse
import ipdb
import time
import math
import cv2
import codecs, json
from tensorflow.contrib.rnn import core_rnn_cell
from sklearn.metrics import average_precision_score
from itertools import cycle
from scipy.misc import imread, imresize

from cnnDoc2Vec import cnnDoc2Vec
from vgg16 import vgg16

#####################################################
#                 Global Parameters		    #  
#####################################################
print 'Loading parameters ...'
# Data input setting
input_img_h5 = './data_img.h5'
input_ques_h5 = './data_prepro.h5'
input_json = './data_prepro.json'
dataPath = './pre-trained/'
# Train Parameters setting
learning_rate = 3e-4			# learning rate for rmsprop
#starter_learning_rate = 3e-4
learning_rate_decay_start = -1		# at what iteration to start decaying learning rate? (-1 = dont)
batch_size = 100			# batch_size for each iterations
input_embedding_size = 300		# he encoding size of each token in the vocabulary
num_output = 1000			# number of output answers
img_norm = 1				# normalize the image feature. 1 = normalize, 0 = not normalize
decay_factor = 0.99997592083
dropoutRate = 0.3
# Check point
checkpoint_path = 'model_save/'
image_dim = 4096
# misc
gpu_id = 0
max_itr = 150000
n_epochs = 300
max_words_q = 24
num_answer = 1000

weight_decay = 0.004

attention_embedding_size = 512


def get_data():

#pre-trained word2vec
	word_vectors = loadmat(dataPath + 'GoogleNews-vectors-negative300.mat')
	word_vectors = word_vectors['vectors']
	with open(dataPath + 'dict.txt', 'r') as filereader:
	    words = filereader.readlines()
	words = [word.lower().strip() for word in words]
	word_vectors = np.vstack((word_vectors, np.zeros([300]))) #None Vec
	#Create reverse dict
	word_to_index = {}
	for i in range(len(words)):
	    word_to_index[words[i]] = i

	dataset = {}
	train_data = {}
	# load json file
	print('loading json file...')
	with open(input_json) as data_file:
		data = json.load(data_file)
	for key in data.keys():
		dataset[key] = data[key]

	# load image feature
	print('loading image feature...')
	with h5py.File(input_img_h5,'r') as hf:
	    # -----0~82459------
	    tem = hf.get('images_train')
	    img_feature = np.array(tem)
	#ipdb.set_trace()
	# load h5 file
	print('loading h5 file...')
	with h5py.File(input_ques_h5,'r') as hf:
	    # total number of training data is 215375
	    # question is (26, )
	    tem = hf.get('ques_train')
	    train_data['question'] = np.array(tem)
	    # max length is 23
	    tem = hf.get('ques_length_train')
	    train_data['length_q'] = np.array(tem)
	    # total 82460 img
	    tem = hf.get('img_pos_train')
		# convert into 0~82459
	    train_data['img_list'] = np.array(tem)-1
	    # answer is 1~1000
	    tem = hf.get('answers')
	    train_data['answers'] = np.array(tem)-1
	
	#print('question aligning')
	#train_data['question'] = right_align(train_data['question'], train_data['length_q'])
	#ipdb.set_trace()
	#generate indexMap
	indexMap = {}
	for k,v in dataset['ix_to_word'].iteritems():
		if v in word_to_index:
			indexMap[int(k)] = word_to_index[v]
		else:
			indexMap[int(k)] = 3000000
	
	prepro_que = train_data['question']
	for i in range(prepro_que.shape[0]):
		for j in range(prepro_que.shape[1]):
			if prepro_que[i][j] in indexMap:
				prepro_que[i][j] = indexMap[prepro_que[i][j]]
			else:
				prepro_que[i][j] = 3000000

	#ipdb.set_trace()
	print('Normalizing image feature')
	if img_norm:
	    tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
	    img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))
	return [dataset, img_feature, train_data, word_vectors, word_to_index, indexMap]


def get_data_test():

#pre-trained word2vec
	word_vectors = loadmat(dataPath + 'GoogleNews-vectors-negative300.mat')
	word_vectors = word_vectors['vectors']
	with open(dataPath + 'dict.txt', 'r') as filereader:
	    words = filereader.readlines()
	words = [word.lower().strip() for word in words]
	word_vectors = np.vstack((word_vectors, np.zeros([300]))) #None Vec
	#Create reverse dict
	word_to_index = {}
	for i in range(len(words)):
	    word_to_index[words[i]] = i

	dataset = {}
	test_data = {}
	# load json file
	print('loading json file...')
	with open(input_json) as data_file:
		data = json.load(data_file)
	for key in data.keys():
		dataset[key] = data[key]

	# load image feature
	print('loading image feature...')
	with h5py.File(input_img_h5,'r') as hf:
	    # -----0~82459------
	    tem = hf.get('images_test')
	    img_feature = np.array(tem)
	#ipdb.set_trace()
	# load h5 file
	print('loading h5 file...')
	with h5py.File(input_ques_h5,'r') as hf:
	    # total number of training data is 215375
	    # question is (26, )
	    tem = hf.get('ques_test')
	    test_data['question'] = np.array(tem)
	    # max length is 23
	    tem = hf.get('ques_length_test')
	    test_data['length_q'] = np.array(tem)
	    # total 82460 img
	    tem = hf.get('img_pos_test')
		# convert into 0~82459
	    test_data['img_list'] = np.array(tem)-1
	    # answer is 1~1000
	    tem = hf.get('answers')
	    test_data['answers'] = np.array(tem)-1
	
	#print('question aligning')
	#train_data['question'] = right_align(train_data['question'], train_data['length_q'])
	#ipdb.set_trace()
	#generate indexMap
	indexMap = {}
	for k,v in dataset['ix_to_word'].iteritems():
		if v in word_to_index:
			indexMap[int(k)] = word_to_index[v]
		else:
			indexMap[int(k)] = 3000000
	
	prepro_que = test_data['question']
	for i in range(prepro_que.shape[0]):
		for j in range(prepro_que.shape[1]):
			if prepro_que[i][j] in indexMap:
				prepro_que[i][j] = indexMap[prepro_que[i][j]]
			else:
				prepro_que[i][j] = 3000000

	#ipdb.set_trace()
	print('Normalizing image feature')
	if img_norm:
	    tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
	    img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))
	return [dataset, img_feature, test_data, word_vectors, word_to_index, indexMap]


def train():
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	sess = tf.Session(config=session_conf)
	print "loading training data"
	[dataset, img_feature, train_data, word_vectors, word_to_index, indexMap] = get_data()
	#ipdb.set_trace()
	_, _, test_data, _, _, _ = get_data_test()
	textCNN = cnnDoc2Vec(max_words_q, input_embedding_size, batch_size, dropoutRate)
	#visualCNN = vgg16()
	imgs = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
	visualCNN = vgg16(imgs, 'pre-trained/vgg16_weights.npz',sess)
	
	#=================Build Fused Model
	#imageFeature = tf.placeholder(tf.float32, [batch_size, image_dim])
	visualFeature = visualCNN.conv5_3 #500 14 14 512
	textFeature = textCNN.conv3 #500 1 6 512
	#textFeature = tf.reshape(textFeature, [batch_size, 512 * 6])
	#fusedFeature = tf.concat([imageFeature, textFeature], 1)
	textFeature = tf.reshape(textFeature, [batch_size, 1 * 6 , 512])
	visualFeature = tf.reshape(visualFeature, [batch_size, 14 * 14, 512])

	V = tf.transpose(visualFeature, [0, 2, 1])
	Q = tf.transpose(textFeature, [0, 2, 1])

	
	#############################Co-Attention Model################################


	with tf.variable_scope('affinity_matrix') as scope:
		try:
			W = tf.get_variable(
				name = "weights",
				shape= [batch_size, 512, 512],
				initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
				dtype=tf.float32)
		except Exception as  e:
			scope.reuse_variables()
			W = tf.get_variable(name = "weights")
		#visualFeature = tf.reshape(visualFeature, [batch_size, -1])
		#textFeature = tf.reshape(textFeature, [batch_size, -1])
		#W = tf.reshape(W, [batch_size, -1])

		C = tf.matmul(Q, W, transpose_a = True)
		C = tf.matmul(C, V)
		C = tf.nn.tanh(C)

	with tf.variable_scope('Hypo') as scope:
		try:
			W_v = tf.get_variable(
				name = "weights_v",
				shape= [batch_size, attention_embedding_size, 512],
				initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
				dtype=tf.float32)
			W_q = tf.get_variable(
				name = "weights_q",
				shape= [batch_size, attention_embedding_size, 512],
				initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
				dtype=tf.float32)
		except Exception as  e:
			scope.reuse_variables()
			W_v = tf.get_variable(name = "weights_v")
			W_q = tf.get_variable(name = "weights_q")

		corelate_v = tf.add(tf.matmul(W_v, V), tf.matmul(tf.matmul(W_q,Q),C))
		H_v = tf.nn.tanh(corelate_v)

		corelate_q = tf.add(tf.matmul(W_q, Q), tf.matmul(tf.matmul(W_v,V),C, transpose_b=True))
		H_q = tf.nn.tanh(corelate_q)

	with tf.variable_scope('attention') as scope:
		try:
			w_hv = tf.get_variable(
				name = "w_hv",
				shape= [batch_size, attention_embedding_size,1],
				initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
				dtype=tf.float32)
			w_hq = tf.get_variable(
				name = "w_hq",
				shape= [batch_size, attention_embedding_size, 1],
				initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
				dtype=tf.float32)
		except Exception as e:
			scope.reuse_variables()
			w_hv = tf.get_variable(name = "w_hv")
			w_hq = tf.get_variable(name = "w_hq")
		a_v = tf.nn.softmax(tf.matmul(w_hv, H_v, transpose_a= True))
		a_q = tf.nn.softmax(tf.matmul(w_hq, H_q, transpose_a= True))

	V_weighted = tf.matmul(V, a_v, transpose_b=True)
	Q_weighted = tf.matmul(Q, a_q, transpose_b=True)

	fusedFeature = tf.concat([V_weighted, Q_weighted], 1)
	fusedFeature = tf.reshape(fusedFeature, [batch_size, 1024])

	# #Fully Connected Layer
	# #fuse_fc1
	# with tf.variable_scope('fuse_fc1') as scope:
	# 	shape = int(np.prod(fusedFeature.get_shape()[1:]))
	# 	try:
	# 		fc1w = tf.get_variable(
	# 			name = "weights",
	# 			shape = [shape, 4096],
	# 			initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
	# 			dtype=tf.float32)
	# 		fc1b = tf.get_variable(
	# 			name = "biases",
	# 			shape = [4096],
	# 			initializer = tf.constant_initializer(0.0, dtype = tf.float32),
	# 			dtype= tf.float32)
	# 	except Exception as e:
	# 		scope.reuse_variables()
	# 		fc1w = tf.get_variable(name = "weights")
	# 		fc1b = tf.get_variable(name = "biases")
	# 	fc1l = tf.nn.bias_add(tf.matmul(fusedFeature, fc1w), fc1b)
	# 	fc1 = tf.nn.relu(fc1l)

	# #fuse_fc2
	# with tf.variable_scope('fuse_fc2') as scope:
	# 	try:
	# 		fc2w = tf.get_variable(
	# 			name = "weights",
	# 			shape = [4096, 4096],
	# 			initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
	# 			dtype=tf.float32)
	# 		fc2b = tf.get_variable(
	# 			name = "biases",
	# 			shape = [4096],
	# 			initializer = tf.constant_initializer(0.0, dtype = tf.float32),
	# 			dtype= tf.float32)
	# 	except Exception as e:
	# 		scope.reuse_variables()
	# 		fc2w = tf.get_variable(name = "weights")
	# 		fc2b = tf.get_variable(name = "biases")
	# 	fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
	# 	fc2 = tf.nn.relu(fc2l)

	# #fuse_fc3_softmax
	with tf.variable_scope('fuse_fc3') as scope:
		shape = int(np.prod(fusedFeature.get_shape()[1:]))
		try:
			fc3w = tf.get_variable(
				name = "weights",
				shape = [1024, 1000],
				initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
				dtype=tf.float32)
			fc3b = tf.get_variable(
				name = "biases",
				shape = [1000],
				initializer = tf.constant_initializer(0.0, dtype = tf.float32),
				dtype= tf.float32)
		except Exception as e:
			scope.reuse_variables()
			fc3w = tf.get_variable(name = "weights")
			fc3b = tf.get_variable(name = "biases")
		_weight_decay = tf.multiply(tf.nn.l2_loss(fc3w), weight_decay, name='weight_loss')
		tf.add_to_collection('losses', _weight_decay)
		fc3l = tf.nn.bias_add(tf.matmul(fusedFeature, fc3w), fc3b)
		#fc3 = tf.nn.relu(fc3l)
	#probs = tf.nn.softmax(fc3l)



	label = tf.placeholder(tf.int32, [batch_size])
	label_one_hot = tf.one_hot(tf.cast(label, dtype = tf.int32), num_answer)
	losses = tf.nn.softmax_cross_entropy_with_logits(logits = fc3l, labels = label_one_hot)
	losses = tf.reduce_mean(losses)
  	loss = tf.add_to_collection('losses', losses)

  	loss_op = tf.add_n(tf.get_collection('losses'), name='total_loss')

	predictions = tf.argmax(tf.nn.softmax(fc3l), 1, name="predictions")
	correct_predictions = tf.equal(predictions, tf.argmax(label_one_hot, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


	

	saver = tf.train.Saver(max_to_keep=100)
	tvars = tf.trainable_variables()
	lr = tf.Variable(learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)

	grads_and_vars = optimizer.compute_gradients(loss_op, tvars)
	#clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in grads_and_vars]
	train_op = optimizer.apply_gradients(grads_and_vars)

	sess.run(tf.initialize_all_variables())
	

	def generateBatches(index):
		
		current_answers = train_data['answers'][index]
		current_img_list = train_data['img_list'][index]
		temp = np.array(dataset['unique_img_train'])
		current_img_files = temp[current_img_list]


		current_img = np.zeros([batch_size, 224, 224, 3])
		for i in range(batch_size):
			current_img[i, :, :] = imresize(imread(current_img_files[i], mode='RGB'), (224, 224))
		#ipdb.set_trace()
		current_question = train_data['question'][index,:max_words_q]
		#generate 4D tensor for CNN

		temp_4d = word_vectors[current_question]
		temp_4d = np.swapaxes(temp_4d, 1, 2)
		temp_4d = temp_4d[:, :, :, np.newaxis]

		current_question = temp_4d

		return current_question, current_answers, current_img
		#temp_4d = current_question[:, np.newaxis, :, np.newaxis]
		#temp_4d = np.repeat(temp_4d, 300, 1)
		
	def generateTestBatches(index):
		#ipdb.set_trace()
		current_answers = test_data['answers'][index]
		current_img_list = test_data['img_list'][index]
		temp = np.array(dataset['unique_img_test'])
		current_img_files = temp[current_img_list]


		current_img = np.zeros([batch_size, 224, 224, 3])
		for i in range(batch_size):
			current_img[i, :, :] = imresize(imread(current_img_files[i], mode='RGB'), (224, 224))

		current_question = test_data['question'][index,:max_words_q]
		#generate 4D tensor for CNN

		temp_4d = word_vectors[current_question]
		temp_4d = np.swapaxes(temp_4d, 1, 2)
		temp_4d = temp_4d[:, :, :, np.newaxis]

		current_question = temp_4d

		return current_question, current_answers, current_img
		#temp_4d = current_question[:, np.newaxis, :, np.newaxis]
		#temp_4d = np.repeat(temp_4d, 300, 1)

	print "start training..."



	for itr in range(max_itr):
		tStart = time.time()
		dataSize = len(train_data['question'])
		batchesPerEpoch = dataSize / batch_size
		shuffle_index = np.random.permutation(dataSize)

		for i in range(batchesPerEpoch):
			index = shuffle_index[np.arange(i * batch_size, min((i + 1) * batch_size, dataSize))]
			current_question, current_answers, current_img = generateBatches(index)
			test_question, test_answers, test_img = generateTestBatches(np.random.random_integers(0, 121512-1, batch_size))
			feed_dict = {
				imgs : current_img,
				textCNN.data_place_holder : current_question,
				label : current_answers
				}
			_, loss_out,acc_out = sess.run(
				[train_op, loss_op, accuracy],
				feed_dict)
			feed_dict = {
				imgs : test_img,
				textCNN.data_place_holder : test_question,
				label : test_answers
				}
			acc_out_test = sess.run(accuracy, feed_dict)
			print "Iteration: ", itr, " Loss: ", loss_out, " Learning Rate: ", lr.eval(session=sess), "acc", acc_out,"test acc", acc_out_test
			#textFeature, fusedFeature, probs, label, label_one_hot = sess.run([textFeature, fusedFeature, probs, label, label_one_hot], feed_dict)
			#ipdb.set_trace()	
		tStop = time.time()

		if np.mod(itr, 100) == 0:
			print "Iteration ", itr, " is done. Saving the model ..."
			saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)
		#current_learning_rate = lr*decay_factor
		#lr.assign(current_learning_rate).eval()
	print "Finally, saving the model ..."
	saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)


train()