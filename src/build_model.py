from collections import namedtuple
import tensorflow as tf 
import pandas as pd 
import numpy as np

def build_simple_lr_nn(numFeatures,hidden_units=10):
	tf.reset_default_graph()
	inputs = tf.placeholder(tf.float32, shape=[None, numFeatures])
	labels = tf.placeholder(tf.float32, shape=[None, 1])
	learning_rate = tf.placeholder(tf.float32)
	is_training = tf.Variable(True, dtype = tf.bool)

	initializer = tf.contrib.layers.xavier_initializer()
	fc = tf.layers.dense(inputs, hidden_units, activation=None, kernel_initializer=initializer)
	fc = tf.layers.batch_normalization(fc, training=is_training)
	fc = tf.nn.relu(fc)

	logits = tf.layers.dense(fc, 1, activation=None)
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
	cost = tf.reduce_mean(cross_entropy)
	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	predicted = tf.nn.sigmoid(logits)
	correct_pred = tf.equal(tf.round(predicted), labels)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	export_nodes = ['inputs', 'labels', 'learning_rate', 'is_training', 'logits','cost', 'optimizer','predicted','accuracy']
	Graph = namedtuple('Graph', export_nodes)
	local_dict = locals()
	graph = Graph(*[local_dict[each] for each in export_nodes])
	return graph