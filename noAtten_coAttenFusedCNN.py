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
batch_size = 500			# batch_size for each iterations
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
	print "loading training data"
	[dataset, img_feature, train_data, word_vectors, word_to_index, indexMap] = get_data()
	#ipdb.set_trace()
	textCNN = cnnDoc2Vec(max_words_q, input_embedding_size, batch_size, dropoutRate)
	#visualCNN = vgg16()
	
	#=================Build Fused Model
	imageFeature = tf.placeholder(tf.float32, [batch_size, image_dim])
	textFeature = textCNN.conv3
	textFeature = tf.reshape(textFeature, [batch_size, 512 * 6])
	fusedFeature = tf.concat([imageFeature, textFeature], 1)

	#Fully Connected Layer
	#fuse_fc1
	with tf.variable_scope('fuse_fc1') as scope:
		shape = int(np.prod(fusedFeature.get_shape()[1:]))
		try:
			fc1w = tf.get_variable(
				name = "weights",
				shape = [shape, 4096],
				initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
				dtype=tf.float32)
			fc1b = tf.get_variable(
				name = "biases",
				shape = [4096],
				initializer = tf.constant_initializer(0.0, dtype = tf.float32),
				dtype= tf.float32)
		except Exception as e:
			scope.reuse_variables()
			fc1w = tf.get_variable(name = "weights")
			fc1b = tf.get_variable(name = "biases")
		_weight_decay = tf.multiply(tf.nn.l2_loss(fc1w), weight_decay, name='weight_loss')
		tf.add_to_collection('losses', _weight_decay)
		fc1l = tf.nn.bias_add(tf.matmul(fusedFeature, fc1w), fc1b)
		fc1 = tf.nn.relu(fc1l)

	#fuse_fc2
	with tf.variable_scope('fuse_fc2') as scope:
		try:
			fc2w = tf.get_variable(
				name = "weights",
				shape = [4096, 4096],
				initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
				dtype=tf.float32)
			fc2b = tf.get_variable(
				name = "biases",
				shape = [4096],
				initializer = tf.constant_initializer(0.0, dtype = tf.float32),
				dtype= tf.float32)
		except Exception as e:
			scope.reuse_variables()
			fc2w = tf.get_variable(name = "weights")
			fc2b = tf.get_variable(name = "biases")
		_weight_decay = tf.multiply(tf.nn.l2_loss(fc2w), weight_decay, name='weight_loss')
		tf.add_to_collection('losses', _weight_decay)
		fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
		fc2 = tf.nn.relu(fc2l)

	#fuse_fc3_softmax
	with tf.variable_scope('fuse_fc3') as scope:
		shape = int(np.prod(fusedFeature.get_shape()[1:]))
		try:
			fc3w = tf.get_variable(
				name = "weights",
				shape = [4096, 1000],
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
		fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
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


	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	sess = tf.Session(config=session_conf)

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
		current_img = img_feature[current_img_list,:]

		current_question = train_data['question'][index,:max_words_q]
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
			feed_dict = {
				imageFeature : current_img,
				textCNN.data_place_holder : current_question,
				label : current_answers
				}
			_, loss_out,acc_out = sess.run(
				[train_op, loss_op, accuracy],
				feed_dict)
			print "Iteration: ", itr, " Loss: ", loss_out, " Learning Rate: ", lr.eval(session=sess), "acc", acc_out
			#textFeature, fusedFeature, probs, label, label_one_hot = sess.run([textFeature, fusedFeature, probs, label, label_one_hot], feed_dict)
			#ipdb.set_trace()	
		tStop = time.time()

		if np.mod(itr, 1000) == 0:
			print "Iteration ", itr, " is done. Saving the model ..."
			saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)
		#current_learning_rate = lr*decay_factor
		#lr.assign(current_learning_rate).eval()
	print "Finally, saving the model ..."
	saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)


def test(model_path='model_save/model-50'):
	print "loading test data"
	[dataset, img_feature, test_data, word_vectors, word_to_index, indexMap] = get_data_test()
	#ipdb.set_trace()
	textCNN = cnnDoc2Vec(max_words_q, input_embedding_size, batch_size, dropoutRate)
	#visualCNN = vgg16()
	
	#=================Build Fused Model
	imageFeature = tf.placeholder(tf.float32, [batch_size, image_dim])
	textFeature = textCNN.conv3
	textFeature = tf.reshape(textFeature, [batch_size, 512 * 6])
	fusedFeature = tf.concat([imageFeature, textFeature], 1)

	#Fully Connected Layer
	#fuse_fc1
	with tf.variable_scope('fuse_fc1') as scope:
		shape = int(np.prod(fusedFeature.get_shape()[1:]))
		try:
			fc1w = tf.get_variable(
				name = "weights",
				shape = [shape, 4096],
				initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
				dtype=tf.float32)
			fc1b = tf.get_variable(
				name = "biases",
				shape = [4096],
				initializer = tf.constant_initializer(0.0, dtype = tf.float32),
				dtype= tf.float32)
		except Exception as e:
			scope.reuse_variables()
			fc1w = tf.get_variable(name = "weights")
			fc1b = tf.get_variable(name = "biases")
		fc1l = tf.nn.bias_add(tf.matmul(fusedFeature, fc1w), fc1b)
		fc1 = tf.nn.relu(fc1l)

	#fuse_fc2
	with tf.variable_scope('fuse_fc2') as scope:
		try:
			fc2w = tf.get_variable(
				name = "weights",
				shape = [4096, 4096],
				initializer = tf.truncated_normal_initializer(stddev = 5e-2, dtype = tf.float32),
				dtype=tf.float32)
			fc2b = tf.get_variable(
				name = "biases",
				shape = [4096],
				initializer = tf.constant_initializer(0.0, dtype = tf.float32),
				dtype= tf.float32)
		except Exception as e:
			scope.reuse_variables()
			fc2w = tf.get_variable(name = "weights")
			fc2b = tf.get_variable(name = "biases")
		fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
		fc2 = tf.nn.relu(fc2l)

	#fuse_fc3_softmax
	with tf.variable_scope('fuse_fc3') as scope:
		shape = int(np.prod(fusedFeature.get_shape()[1:]))
		try:
			fc3w = tf.get_variable(
				name = "weights",
				shape = [4096, 1000],
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
		fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
		#fc3 = tf.nn.relu(fc3l)
	probs = tf.nn.softmax(fc3l)

	label = tf.placeholder(tf.int32, [batch_size])
	label_one_hot = tf.one_hot(tf.cast(label, dtype = tf.int32), num_answer)
	losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = fc3l, labels = label)
	loss_op = tf.reduce_mean(losses)


	predictions = tf.argmax(probs, 1, name="predictions")
	correct_predictions = tf.equal(predictions, tf.argmax(label_one_hot, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	sess = tf.Session(config=session_conf)

	saver = tf.train.Saver()
	saver.restore(sess, model_path)	

	def generateBatches(index):
		
		current_answers = test_data['answers'][index]
		current_img_list = test_data['img_list'][index]
		current_img = img_feature[current_img_list,:]

		current_question = test_data['question'][index,:max_words_q]
		#generate 4D tensor for CNN

		temp_4d = word_vectors[current_question]
		temp_4d = np.swapaxes(temp_4d, 1, 2)
		temp_4d = temp_4d[:, :, :, np.newaxis]

		current_question = temp_4d

		return current_question, current_answers, current_img
		#temp_4d = current_question[:, np.newaxis, :, np.newaxis]
		#temp_4d = np.repeat(temp_4d, 300, 1)
		

	print "start testing..."




	dataSize = len(test_data['question'])
	batchesPerEpoch = dataSize / batch_size
	#shuffle_index = np.random.permutation(dataSize)
	indices_set = np.arange(dataSize)

	for i in range(batchesPerEpoch):
		index = indices_set[np.arange(i * batch_size, min((i + 1) * batch_size, dataSize))]
		if len(index) < batch_size:
			break
		current_question, current_answers, current_img = generateBatches(index)
		feed_dict = {
			imageFeature : current_img,
			textCNN.data_place_holder : current_question,
			label : current_answers
			}
		print sess.run(accuracy,feed_dict)
		#textFeature, fusedFeature, probs, label, label_one_hot = sess.run([textFeature, fusedFeature, probs, label, label_one_hot], feed_dict)
		#ipdb.set_trace()	




train()