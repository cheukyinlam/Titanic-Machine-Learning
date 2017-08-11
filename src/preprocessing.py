import pandas as pd
from preprocessing_util import *

def processing_for_both(data):

	# fill blank entries in the following columns 
	nan_columns = ["Age", "SibSp", "Parch"]
	data = nan_padding(data, nan_columns)

	# drop columns
	not_concerned_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]
	data = drop_not_concerned(data, not_concerned_columns)

	# encode class to onehot
	class_columns = ["Pclass"]
	data = class_to_onehot(data, class_columns)

	# encode sex as 1 and 0
	data = sex_to_int(data)

	# scale age
	data = normalize_age(data)

	return data

def preprocess_training_set(train_path):
	data = pd.read_csv(train_path)
	data = processing_for_both(data)

	# split validation set from training set
	train_x, train_y, valid_x, valid_y = split_valid_test_data(data)
	return train_x, train_y, valid_x, valid_y

def preprocess_test_set(test_path):
	data = pd.read_csv(test_path)
	PassengerId = data["PassengerId"]
	data = processing_for_both(data)
	return data, PassengerId

# def preprocess(train_path, test_path):
# 	# read files
# 	train_data 	= pd.read_csv(train_path)
# 	test_data	= pd.read_csv(test_path)

# 	# fill blank entries in the following columns 
# 	nan_columns = ["Age", "SibSp", "Parch"]

# 	train_data = nan_padding(train_data, nan_columns)
# 	test_data = nan_padding(test_data, nan_columns)

# 	test_passenger_id = test_data["PassengerId"]

# 	# drop columns
# 	not_concerned_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]

# 	train_data = drop_not_concerned(train_data, not_concerned_columns)
# 	test_data = drop_not_concerned(test_data, not_concerned_columns)

# 	# encode class to onehot
# 	class_columns = ["Pclass"]
# 	train_data = class_to_onehot(train_data, class_columns)
# 	test_data = class_to_onehot(test_data, class_columns)

# 	# encode sex as 1 and 0
# 	train_data = sex_to_int(train_data)
# 	test_data = sex_to_int(test_data)

# 	# scale age
# 	train_data = normalize_age(train_data)
# 	test_data = normalize_age(test_data)

# 	# split validation set from training set
# 	train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)
# 	return train_x, train_y, valid_x, valid_y, test_data, test_passenger_id
