import tensorflow as tf 
import pandas as pd 
import numpy as np
from preprocessing import preprocess_training_set
from build_model import build_simple_lr_nn

train_path 	= "../data/train.csv"

train_x, train_y, valid_x, valid_y = preprocess_training_set(train_path)

model = build_simple_lr_nn(train_x.shape[1])

def get_batch(data_x, data_y, batch_size=32):
	batch_n = len(data_x)
	for i in range(batch_n):
		batch_x = data_x[i*batch_size:(i+1)*batch_size]
		batch_y = data_y[i*batch_size:(i+1)*batch_size]
		yield batch_x, batch_y

epochs = 200
train_collect = 50
train_print = train_collect * 2

learning_rate_value = 0.001
batch_size = 16

x_collect = []
train_loss_collect = []
train_acc_collect = []
valid_loss_collect = []
valid_acc_collect = []

saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	iteration = 0
	for e in range(epochs):
		for batch_x, batch_y in get_batch(train_x, train_y, batch_size):
			iteration += 1
			feed = {
				model.inputs: train_x,
				model.labels: train_y,
				model.learning_rate:learning_rate_value,
				model.is_training:True
			}

			train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)
			if iteration % train_collect == 0:
				x_collect.append(e)
				train_loss_collect.append(train_loss)
				train_acc_collect.append(train_acc)
				if iteration % train_print == 0:
					print("Epoch = {}/{}".format(e+1, epochs),
						"Train Loss: {:.4f}".format(train_loss),
						"Train Acc: {:.4f}".format(train_acc)
					)
				feed = {
					model.inputs: valid_x,
					model.labels: valid_y,
					model.is_training:False
				}
				val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
				valid_loss_collect.append(val_loss)
				valid_acc_collect.append(val_acc)

				if iteration % train_print == 0:
					print("Epoch: {}/{}".format(e+1, epochs),
						"Validation Loss: {:.4f}".format(val_loss),
						"Validation Acc: {:.4}".format(val_acc)
						)

	saver.save(sess, "./titanic.ckpt")

