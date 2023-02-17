import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#%% test drop out; in prediction, training=False argument does not make difference
inputs = tf.keras.Input(shape=(1,))
x = tf.keras.layers.Dense(20, activation='relu')(inputs)
x = tf.keras.layers.Dense(20, activation='relu')(x)
x = tf.keras.layers.Dense(20, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1)(x)
outputs = tf.squeeze(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

x = tf.range(-5,5,0.01)
y = tf.math.sin(x)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss = tf.losses.MeanSquaredError())
model.fit(x[:,tf.newaxis], y, epochs=100)

# prediction
pred1 = model(x[:,tf.newaxis])
pred2 = model(x[:,tf.newaxis], training=False)

plt.plot(x,y,label='true')
plt.plot(x,pred1, label='without false')
plt.plot(x,pred2, label='with false')
plt.legend()

#%% test padding
raw = [[1,2],[2,3,4]]
x = tf.keras.preprocessing.sequence.pad_sequences(raw, padding='post')
mask=tf.cast(tf.not_equal(x,0), tf.float32)

#%% lambda layer to create look ahead mask
look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

x = tf.constant([[2,81]])
create_look_ahead_mask(x.shape[1])