import pickle
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout

# read data
with open('E:/datasets/cifar-10-python/cifar-10-batches-py/data_batch_1', 'rb') as f:
	train_dict = pickle.load(f, encoding='bytes')

with open('E:/datasets/cifar-10-python/cifar-10-batches-py/test_batch', 'rb') as f:
	test_dict = pickle.load(f, encoding='bytes')

BATCH_SIZE = 32	
BUFFER = 10000
LEARNING_RATE = 1e-5


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

def vgg_net():
	'''16 layers'''
	inputs = tf.keras.Input(shape=(32, 32, 3))

	x = Conv2D(filters = 64, kernel_size = 3, padding='same', activation='relu')(inputs)
	x = Conv2D(filters = 64, kernel_size = 3, padding='same', activation='relu')(x)
	x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x) # size = 16

	x = Conv2D(filters = 128, kernel_size = 3, padding='same', activation='relu')(x)
	x = Conv2D(filters = 128, kernel_size = 3, padding='same', activation='relu')(x)
	x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x) # size = 8

	x = Conv2D(filters = 256, kernel_size = 3, padding='same', activation='relu')(x)
	x = Conv2D(filters = 256, kernel_size = 3, padding='same', activation='relu')(x)
	x = Conv2D(filters = 256, kernel_size = 3, padding='same', activation='relu')(x)
	x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x) # size = 4

	x = Conv2D(filters = 512, kernel_size = 3, padding='same', activation='relu')(x)
	x = Conv2D(filters = 512, kernel_size = 3, padding='same', activation='relu')(x)
	x = Conv2D(filters = 512, kernel_size = 3, padding='same', activation='relu')(x)
	x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'same')(x) # size = 2

	x = Conv2D(filters = 512, kernel_size = 3, padding='same', activation='relu')(x)
	x = Conv2D(filters = 512, kernel_size = 3, padding='same', activation='relu')(x)
	x = Conv2D(filters = 512, kernel_size = 3, padding='same', activation='relu')(x)
	x = MaxPooling2D(pool_size = 2, strides = 2, padding = 'valid')(x) # size = 1

	x = tf.reshape(x, [-1, 512]) # [batch_size, 512]

	x = Dense(units=256, activation = 'relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(units=256, activation = 'relu')(x)
	x = Dropout(0.5)(x)

	outputs = Dense(10)(x)

	return tf.keras.Model(inputs=inputs, outputs=outputs)

model = vgg_net()
optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
model.compile(loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
			  optimizer = optimizer)
model.fit(train_dataset, 
	validation_data = test_dataset, 
	epochs=10, 
	use_multiprocessing=True)

acc = 0
for x,y in train_dataset.take(10000 // 32 + 1):
	pred = tf.cast(tf.argmax(model(x), axis=1), tf.int32)
	y = tf.cast(y, tf.int32)
	acc += tf.reduce_mean(tf.cast(pred == y, tf.float32))
print('train accuracy {}'.format(acc / (10000 // 32 + 1)))


