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
import os
import tensorflow as tf



def train(trainX, trainY):
	no_of_images=trainX.shape[0]
	hidden_layer_size=100
	input_layer_size=784
	output_layer_size=10
	batch_sz=100
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1),name="W")
	b = tf.Variable(tf.constant(0.1, shape=[100]),name="b")
	W2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1),name="W2")
	b2 = tf.Variable(tf.constant(0.1, shape=[10]),name="b2")
	#W = tf.Variable(tf.zeros([784, 10]),name="W")
	#b = tf.Variable(tf.zeros([10]),name="b")
	h=tf.nn.relu(tf.matmul(x, W) + b)
	y = tf.matmul(h, W2) + b2
	y_ = tf.placeholder(tf.float32, [None, 10])
	#saver = tf.train.Saver()
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	#cross_entropy=tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b)
	train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
	sess = tf.Session()
	#sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	for j in range(10):
		for i in range(0,no_of_images,batch_sz):
			input=np.ones(batch_sz*(input_layer_size)).reshape(batch_sz,input_layer_size)
			input[:,:] = trainX[i : i+batch_sz].reshape(batch_sz,input_layer_size)
			target=np.zeros(10*batch_sz).reshape(batch_sz, 10) 	
			for k in range(0,batch_sz):
				target[k][trainY[i+k]]=1.0
			sess.run(train_step, feed_dict={x: input, y_: target})
	#save_path = saver.save(sess, "C:/Users/Rishabh/Desktop/DeepLearning/assign3/files/my-model")
	tf.add_to_collection('vars', W)
	tf.add_to_collection('vars', b)
	tf.add_to_collection('vars', W2)
	tf.add_to_collection('vars', b2)
	saver = tf.train.Saver()
	save_path = saver.save(sess, os.path.join(os.getcwd(), 'weight1/model.ckpt'))
	v_ = sess.run(W)
	print(v_)
	print("asdfaaaaaaaaaaaaaaaaaaaaaaaaadd")
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy,{x: input, y_: target}))
	'''
	Complete this function.
	'''


def test(testX):
	no_of_images=testX.shape[0]
	hidden_layer_size=100
	input_layer_size=784
	output_layer_size=10
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1),name="W")
	b = tf.Variable(tf.constant(0.1, shape=[100]),name="b")
	W2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1),name="W2")
	b2 = tf.Variable(tf.constant(0.1, shape=[10]),name="b2")
	sess = tf.InteractiveSession()
	init = tf.global_variables_initializer()
	sess.run(init)
	#v_ = sess.run(W)
	#print(v_)
	#print("safasdfs")
	new_saver = tf.train.Saver()
	new_saver.restore(sess, os.path.join(os.getcwd(), 'weight1/model.ckpt'))
	all_vars = tf.get_collection('vars')
	ik=0
	for v in all_vars:
		if ik==0:
			W = v
			ik=1
		elif ik==1:
			b=v
			ik=2
		elif ik==2:
			W2=v
			ik=3
		elif ik==3:
			b2=v
			ik=4
	#saver = tf.train.Saver()
	#saver.restore(sess, os.path.join(os.getcwd(), 'model.ckpt'))
	#y_ = tf.placeholder(tf.float32, [None, 10])
	v_ = sess.run(W)
	print(v_)
	print("safasdfs")
	h=tf.nn.relu(tf.matmul(x, W) + b)
	y = tf.matmul(h, W2) + b2
	y_ = tf.placeholder(tf.float32, [None, 10])
	prediction=tf.argmax(y,1)
	input=np.ones(no_of_images*(input_layer_size)).reshape(no_of_images,input_layer_size)
	input[:,:] = testX[0 : no_of_images].reshape(no_of_images,input_layer_size)
	result=sess.run(prediction,{x: input})
	'''
	Complete this function.
	This function must read the weight files and
	return the predicted labels.
	The returned object must be a 1-dimensional numpy array of
	length equal to the number of examples. The i-th element
	of the array should contain the label of the i-th test
	example.
	'''
	
	return result
	