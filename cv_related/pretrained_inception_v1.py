import tensorflow as tf 
import pickle 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, AvgPool2D
import numpy as np 
from tensorflow.keras.applications import inception_v3

# read data
with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_1', 'rb') as f:
	train_dict_1 = pickle.load(f, encoding='bytes')

with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_2', 'rb') as f:
	train_dict_2 = pickle.load(f, encoding='bytes')

with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_3', 'rb') as f:
	train_dict_3 = pickle.load(f, encoding='bytes')

with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_4', 'rb') as f:
	train_dict_4 = pickle.load(f, encoding='bytes')

with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_5', 'rb') as f:
	train_dict_5 = pickle.load(f, encoding='bytes')

with open('E:/datasets/cifar-10-python/cifar-10-batches-py/test_batch', 'rb') as f:
	test_dict = pickle.load(f, encoding='bytes')

BATCH_SIZE = 32	
BUFFER = 10000
LEARNING_RATE = 1e-5
EPOCHS = 10

def preprocess(x, y):
	'''
	INPUT
		x = [batch_size, 32 * 32 * 3], y = [batch_size,]
	OUTPUT
		x = [batch_size, 32, 32, 3], resize to [-1, 1]
	'''
	x = tf.cast(x, tf.float32)
	x = x / 255.0 * 2.0 - 1.0 # change data scale to [-1, 1]

	x = tf.reshape(x, [-1, 3, 32, 32])
	x = tf.transpose(x, [0, 2, 3, 1])
	return (x, y)

train_x = np.concatenate([train_dict_1[b'data'], train_dict_2[b'data'], train_dict_3[b'data'],
	train_dict_4[b'data'], train_dict_5[b'data']], axis=0)
train_y = np.concatenate([train_dict_1[b'labels'], train_dict_2[b'labels'], train_dict_3[b'labels'],
	train_dict_4[b'labels'], train_dict_5[b'labels']], axis=0)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.shuffle(BUFFER).batch(BATCH_SIZE)
train_dataset = train_dataset.map(preprocess)

test_dataset = tf.data.Dataset.from_tensor_slices((test_dict[b'data'], test_dict[b'labels']))
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(preprocess)

def inception_net(w):
	inputs = tf.keras.Input(shape=(32,32,3))
	x = w.get_layer(index=1)(inputs)
	for i in range(2, len(w.layers) - 2):
		x = w.get_layer(index=i)(x)

	x = Dense(256, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(256, activation='relu')(x)
	x = Dropout(0.5)(x)
	outputs = Dense(10)(x)

	return tf.keras.Model(inputs=inputs, outputs=outputs)

model = inception_net(inception_v3.InceptionV3())