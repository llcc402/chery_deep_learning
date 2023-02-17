import tensorflow as tf 
import pickle 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.keras.layers import AvgPool2D, BatchNormalization, Flatten
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

BATCH_SIZE = 128
BUFFER = 10000
LEARNING_RATE = 1e-3
EPOCHS = 60

train_x = np.concatenate([train_dict_1[b'data'], train_dict_2[b'data'], train_dict_3[b'data'],
	train_dict_4[b'data'], train_dict_5[b'data']], axis=0)
train_y = np.concatenate([train_dict_1[b'labels'], train_dict_2[b'labels'], train_dict_3[b'labels'],
	train_dict_4[b'labels'], train_dict_5[b'labels']], axis=0)

train_x = train_x / 255.0 * 2 - 1
test_x = test_dict[b'data'] / 255.0 * 2 - 1 
test_y = test_dict[b'labels']

train_y = tf.cast(train_y, tf.int64)
test_y = tf.cast(test_y, tf.int64)

train_x = tf.reshape(train_x, [-1, 3, 32, 32])
train_x = tf.transpose(train_x, [0, 2, 3, 1])
test_x = tf.reshape(test_x, [-1, 3, 32, 32])
test_x = tf.transpose(test_x, [0, 2, 3, 1])

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
	rescale=None,
	rotation_range=10,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.1,
	zoom_range=0.1,
	horizontal_flip=False,
	fill_mode='nearest'
	)

def conv_net():
	inputs = tf.keras.Input(shape=(32,32,3))

	x = Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu')(inputs)
	x = BatchNormalization()(x)
	x = Dropout(0.3)(x)

	x = Conv2D(filters=64, kernel_size=3, padding='same', strides=1, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.3)(x)

	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.3)(x)

	x = Conv2D(filters=128, kernel_size=3, padding='same', strides=1, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.3)(x)

	x = MaxPooling2D(pool_size=2)(x)

	x = Flatten()(x)

	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.3)(x)
	x = BatchNormalization()(x)

	x = Dense(256, activation='relu')(x)
	x = Dropout(0.5)(x)

	outputs = Dense(10)(x)

	return tf.keras.Model(inputs=inputs, outputs=outputs)

model = conv_net()
optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
model.compile(loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
			  optimizer = optimizer,
			  metrics=['accuracy'])
model.fit(datagen.flow(train_x, train_y, batch_size=BATCH_SIZE), 
	validation_data = (test_x, test_y), 
	epochs=EPOCHS)