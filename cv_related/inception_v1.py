import tensorflow as tf 
import pickle 

import pickle
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, AvgPool2D

# read data
with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_1', 'rb') as f:
	train_dict = pickle.load(f, encoding='bytes')

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

train_dataset = tf.data.Dataset.from_tensor_slices((train_dict[b'data'], train_dict[b'labels']))
train_dataset = train_dataset.shuffle(BUFFER).batch(BATCH_SIZE)
train_dataset = train_dataset.map(preprocess)

test_dataset = tf.data.Dataset.from_tensor_slices((test_dict[b'data'], test_dict[b'labels']))
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(preprocess)

class InceptionModule(tf.keras.layers.Layer):
	def __init__(self, c1_num_1, c1_num_2, c1_num_3, c1_num_4, c3_num, c5_num):
		'''
		INPUT
			c1_num_1      1*1 conv, number of filters, 
			c1_num_2      1*1 conv, number of filters, before 3*3 conv
			c1_num_3      1*1 conv, number of filters, before 5*5 conv
			c1_num_4      1*1 conv, number of filters, after max pooling
			c3_num        3*3 conv, number of filters, 
			c5_num        5*5 conv, number of filters
		'''
		super(InceptionModule, self).__init__()
		self.c1_layer_1 = Conv2D(filters=c1_num_1, kernel_size=1, padding='same', activation='relu')
		self.c1_layer_2 = Conv2D(filters=c1_num_2, kernel_size=1, padding='same', activation='relu')
		self.c1_layer_3 = Conv2D(filters=c1_num_3, kernel_size=1, padding='same', activation='relu')
		self.c1_layer_4 = Conv2D(filters=c1_num_4, kernel_size=1, padding='same', activation='relu')
		self.c3_layer = Conv2D(filters=c3_num, kernel_size=3, padding='same', activation='relu')
		self.c5_layer = Conv2D(filters=c5_num, kernel_size=5, padding='same', activation='relu')
		self.max_pooling_layer = MaxPooling2D(pool_size=3, strides=1, padding='same')

	def call(self, inputs):
		x1 = self.c1_layer_1(inputs) # [batch_size, width, height, c1_num_1]
		x2 = self.c1_layer_2(inputs) # [batch_size, width, height, c1_num_2]
		x3 = self.c1_layer_3(inputs) # [batch_size, width, height, c1_num_3]
		x4 = self.max_pooling_layer(inputs) # [batch_size, width, height, input_channal_num]

		x2 = self.c3_layer(x2) # [batch_size, width, height, c3_num]
		x3 = self.c5_layer(x3) # [batch_size, width, height, c5_num]
		x4 = self.c1_layer_4(x4) # [batch_size, width, height, c1_num_4]

		outputs = tf.concat([x1, x2, x3, x4], axis=-1)

		return outputs 

def inception_net():
	inputs = tf.keras.Input(shape=(32,32,3))

	x = Conv2D(filters=64, kernel_size=7, padding='same', strides=1, activation='relu')(inputs) #32
	x = MaxPooling2D(pool_size=3, strides=1, padding='same')(x) # 32
	x = InceptionModule(64, 96, 16, 32, 128, 32)(x)
	x = InceptionModule(128, 128, 32, 64, 192, 96)(x)
	x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x) # 16
	x = InceptionModule(192, 96, 16, 64, 208, 48)(x)
	x = InceptionModule(160, 112, 24, 64, 224, 64)(x) # 16
	x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x) # 8
	x = InceptionModule(256, 160, 32, 128, 320, 128)(x) # 8
	x = InceptionModule(384, 192, 48, 128, 384, 128)(x) # 8
	x = AvgPool2D(pool_size=8, strides=1, padding='valid')(x) # 1
	x = tf.reshape(x, [-1, 1024])
	x = Dropout(0.4)(x)
	outputs = Dense(10)(x)

	return tf.keras.Model(inputs = inputs, outputs = outputs)


model = inception_net()
optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
model.compile(loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
			  optimizer = optimizer)
model.fit(train_dataset, 
	validation_data = test_dataset, 
	epochs=EPOCHS, 
	use_multiprocessing=True)