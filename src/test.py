import tensorflow as tf 
import pandas as pd 
import numpy as np
from preprocessing import preprocess_test_set
from build_model import build_simple_lr_nn

test_path	= "../data/test.csv"

test_x, test_passenger_id = preprocess_test_set(test_path)
model = build_simple_lr_nn(test_x.shape[1])
restorer = tf.train.Saver()
with tf.Session() as sess:
	restorer.restore(sess, "./titanic.ckpt")
	feed = {
		model.inputs:test_x,
		model.is_training:False
	}
	test_predict = sess.run(model.predicted, feed_dict=feed)

from sklearn.preprocessing import Binarizer 
binarizer = Binarizer(0.5)
test_predict_result = binarizer.fit_transform(test_predict)
test_predict_result = test_predict_result.astype(np.int32)

evaluation = test_passenger_id.to_frame()
evaluation["Survived"] = test_predict_result

evaluation.to_csv("../output/evaluation_submission.csv",index=False)