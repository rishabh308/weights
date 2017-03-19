'''
Deep Learning Programming Assignment 2
--------------------------------------
Name:
Roll No.:

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf
import os

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
  
def train(trainX, trainY):
	no_of_images=trainX.shape[0]
	input_layer_size=784
	batch_sz=100
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1),name="W_conv1")
	b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]),name="b_conv1")
	x_image = tf.reshape(x, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1),name="W_conv2")
	b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]),name="b_conv2")

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1),name="W_fc1")
	b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]),name="b_fc1")

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1),name="W_fc2")
	b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]),name="b_fc2")

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	for i in range(8):
		for i in range(0,no_of_images,batch_sz):
			input=np.ones(batch_sz*(input_layer_size)).reshape(batch_sz,input_layer_size)
			input[:,:] = trainX[i : i+batch_sz].reshape(batch_sz,input_layer_size)
			target=np.zeros(10*batch_sz).reshape(batch_sz, 10)
			for k in range(0,batch_sz):
				target[k][trainY[i+k]]=1.0
			sess.run(train_step, feed_dict={x: input, y_: target,keep_prob: 0.5})
	tf.add_to_collection('vars', W_conv1)
	tf.add_to_collection('vars', b_conv1)
	tf.add_to_collection('vars', W_conv2)
	tf.add_to_collection('vars', b_conv2)
	tf.add_to_collection('vars', W_fc1)
	tf.add_to_collection('vars', b_fc1)
	tf.add_to_collection('vars', W_fc2)
	tf.add_to_collection('vars', b_fc2)
	saver = tf.train.Saver()
	save_path = saver.save(sess, os.path.join(os.getcwd(), 'weight2/model2.ckpt'))
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("train accuracy %g"%accuracy.eval(feed_dict={x: input, y_: target, keep_prob: 1.0}))

def test(testX):
	no_of_images=testX.shape[0]
	input_layer_size=784
	batch_sz=100
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1),name="W_conv1")
	b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]),name="b_conv1")
	x_image = tf.reshape(x, [-1,28,28,1])
	W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1),name="W_conv2")
	b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]),name="b_conv2")

	W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1),name="W_fc1")
	b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]),name="b_fc1")

	W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1),name="W_fc2")
	b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]),name="b_fc2")

	sess = tf.InteractiveSession()
	init = tf.global_variables_initializer()
	sess.run(init)
	#v_ = sess.run(W)
	#print(v_)
	#print("safasdfs")
	new_saver = tf.train.Saver()
	new_saver.restore(sess, os.path.join(os.getcwd(), 'weight2/model2.ckpt'))
	all_vars = tf.get_collection('vars')
	ik=0
	for v in all_vars:
		if ik==0:
			W_conv1 = v
			ik=1
		elif ik==1:
			b_conv1=v
			ik=2
		elif ik==2:
			W_conv2=v
			ik=3
		elif ik==3:
			b_conv2=v
			ik=4
		elif ik==4:
			W_fc1=v
			ik=5
		elif ik==5:
			b_fc1=v
			ik=6
		elif ik==6:
			W_fc2=v
			ik=7
		elif ik==7:
			b_fc2=v
			ik=8
	#saver = tf.train.Saver()
	#saver.restore(sess, os.path.join(os.getcwd(), 'model.ckpt'))
	#y_ = tf.placeholder(tf.float32, [None, 10])
	#v_ = sess.run(W)
	#print(v_)
	print("safasdfs")
	#y = tf.matmul(x, W) + b
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	y_ = tf.placeholder(tf.float32, [None, 10])
	prediction=tf.argmax(y_conv,1)
	input=np.ones(no_of_images*(input_layer_size)).reshape(no_of_images,input_layer_size)
	input[:,:] = testX[0 : no_of_images].reshape(no_of_images,input_layer_size)
	result=sess.run(prediction,{x: input,keep_prob: 1.0})
	return result
	
