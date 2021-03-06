import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
# fill missing values with mean by default
def nan_padding(data, columns):
    for column in columns:
        imputer=Imputer()
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data


# drop unnecessary columns
def drop_not_concerned(data, columns):
	return data.drop(columns, axis=1)


# change a column with n distinct values to n columns with binary values
def class_to_onehot(data, columns):
	for column in columns:
		data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis = 1)
		data = data.drop(column, axis=1)
	return data


from sklearn.preprocessing import LabelEncoder
# change labels into numbers
def sex_to_int(data):
	le = LabelEncoder()
	le.fit(["male", "female"])
	data["Sex"] = le.transform(data["Sex"])
	return data


from sklearn.preprocessing import MinMaxScaler
# scales the feature (age)
def normalize_age(data):
	scaler = MinMaxScaler()
	data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
	return data

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# extract validation set from training set
def split_valid_test_data(data, fraction=(1-0.8)):
	data_y = data["Survived"]
	lb = LabelBinarizer()
	data_y = lb.fit_transform(data_y)
	data_x = data.drop(["Survived"], axis=1)
	train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size = fraction)
	return train_x.values, train_y, valid_x.values, valid_y