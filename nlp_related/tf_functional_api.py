import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#%% model
inputs = tf.keras.Input(shape = (1,))
x = tf.keras.layers.Dense(10, activation='relu')(inputs)
pred = tf.squeeze(tf.keras.layers.Dense(1)(x))
model = tf.keras.Model(inputs=inputs, outputs=pred, name='sine')

#%% data
x = tf.range(-1,1, 0.01)
y = tf.math.sin(x)
inputs = x[:,tf.newaxis]

#%% fit model with data
model.compile(loss = tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(1e-3))
model.fit(inputs,y, epochs=100)

#%% plot
plt.style.use('seaborn')
plt.plot(x,y, label='true')
plt.plot(x,model.predict(inputs), label='pred')
plt.legend()