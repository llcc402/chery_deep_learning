import tensorflow as tf 
import numpy as np 
from helper_funcs import read_cifar, normalize_cifar, reshape_cifar
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import vgg16 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

train_x, train_y, test_x, test_y = read_cifar()
train_x, test_x = normalize_cifar(train_x, test_x)
train_x, test_x = reshape_cifar(train_x, test_x)

datagen = ImageDataGenerator(horizontal_flip=True)

def conv_net(w):
	inputs = tf.keras.Input(shape=(32,32,3))

	x = w.get_layer(index=1)(inputs)
	for i in range(2,len(w.layers) - 3):
		x = w.get_layer(index=i)(x)

	x = Dense(1024,activation='relu')(x)
	x = Dropout(0.4)(x)
	x = Dense(256, activation='relu')(x)
	x = Dropout(0.4)(x)
	outputs = Dense(10)(x)

	return tf.keras.Model(inputs=inputs, outputs=outputs)

model = conv_net(vgg16.VGG16())

for i in range(len(model.layers) - 5):
	model.layers[i].trainable=False

LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 20

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.1)

model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
	optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
	metrics=['accuracy'])

model.fit(datagen.flow(train_x, train_y, batch_size=BATCH_SIZE),
	validation_data=(test_x, test_y),
	epochs=EPOCHS,
	callbacks=[reduce_lr])

# fine tune pretrained layers
for i in range(len(model.layers) - 10, len(model.layers)):
	model.layers[i].trainable=True     

LEARNING_RATE = 1e-4
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
	optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
	metrics=['accuracy'])

model.fit(datagen.flow(train_x, train_y, batch_size=BATCH_SIZE),
	validation_data=(test_x, test_y),
	epochs=EPOCHS,
	callbacks=[reduce_lr])