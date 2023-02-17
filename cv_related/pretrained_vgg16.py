import pickle
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization 
from tensorflow.keras.applications import vgg16
import numpy as np 
import tensorflow_probability as tfp

BATCH_SIZE = 128
BUFFER = 10000
LEARNING_RATE = 1e-3
EPOCHS = 20

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

def vgg_net(w):
	'''
	INPUT
		w    a pretrained model 
	'''
	inputs = tf.keras.Input(shape=(32, 32, 3))

	x = w.get_layer(index=1)(inputs)
	for i in range(2, len(w.layers) - 3):
		x = w.get_layer(index=i)(x)

	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(256, activation='relu')(x)
	x = Dropout(0.5)(x)
	outputs = Dense(10)(x)

	return tf.keras.Model(inputs=inputs, outputs=outputs)

def color_jitter(x, y):
	x = tf.cast(x, tf.int32)
	x = tf.reshape(x, [3, -1]) # [3, width * height]
	x = tf.transpose(x) # [width * heigth, 3]

	img = tf.cast(x, tf.float32) / 255.0
	img = img - tf.reduce_mean(img, axis=0)
	img = img / tf.math.reduce_std(img, axis=0)

	c = tfp.stats.covariance(img)
	val, vec = tf.linalg.eig(c)
	val = tf.cast(val, tf.float32)
	vec = tf.cast(vec, tf.float32)
	a = tf.random.normal(shape=[3,], stddev=0.1)

	jitter = tf.matmul(vec, (val * a)[:,tf.newaxis])
	jitter = tf.cast(jitter * 255, tf.int32)
	jitter = tf.squeeze(jitter) 

	x1 = tf.clip_by_value(x + jitter, 0, 255) # [width * height, 3]

	x1 = tf.transpose(x1, [1,0]) # [3, width * height]
	x1 = tf.reshape(x1, [3,32,32]) # [3, width, height]
	x1 = tf.transpose(x1, [1,2,0]) # [width, height, 3]

	if tf.random.normal(shape=[1,]) > 0:
		x1 = tf.image.flip_left_right(x1)

	# rescale to [-1,1]
	x1 = tf.cast(x1, tf.float32) / 255.0
	x1 = x1 * 2 - 1 

	return x1, y

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.map(color_jitter)
train_dataset = train_dataset.shuffle(BUFFER).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_dict[b'data'], test_dict[b'labels']))
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(preprocess)


model = vgg_net(vgg16.VGG16())
for i in range(len(model.layers) - 5):
	model.layers[i].trainable=False

optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
model.compile(loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
			  optimizer = optimizer,
			  metrics=['accuracy'])
model.fit(train_dataset, 
	validation_data = test_dataset, 
	epochs=EPOCHS, 
	use_multiprocessing=True)

acc = 0
for x,y in train_dataset.take(train_x.shape[0] // BATCH_SIZE + 1):
	pred = tf.cast(tf.argmax(model(x), axis=1), tf.int32)
	y = tf.cast(y, tf.int32)
	acc += tf.reduce_mean(tf.cast(pred == y, tf.float32))
print('train accuracy {}'.format(acc / (train_x.shape[0] // BATCH_SIZE + 1)))

acc = 0
for x,y in test_dataset.take(10000 // BATCH_SIZE + 1):
	pred = tf.cast(tf.argmax(model(x), axis=1), tf.int32)
	y = tf.cast(y, tf.int32)
	acc += tf.reduce_mean(tf.cast(pred == y, tf.float32))
print('train accuracy {}'.format(acc / (10000 // BATCH_SIZE + 1)))