import tensorflow as tf 
import matplotlib.pyplot as plt

#---------------------------------- model 1 -------------------------------
# x = tf.range(-5, 5, 0.01)
# y = tf.math.sin(x)

# x1 = x[tf.newaxis,:,tf.newaxis]
# y1 = y[tf.newaxis,:,tf.newaxis]

# class SinModel(tf.keras.Model):
# 	def __init__(self):
# 		super(SinModel, self).__init__()
# 		self.W1 = tf.keras.layers.Dense(10)
# 		self.W2 = tf.keras.layers.LSTM(units=10, return_sequences=True)
# 		self.W3 = tf.keras.layers.Dense(1)

# 	def call(self, inputs):
# 		x = self.W1(inputs)
# 		x = self.W2(x)
# 		outputs = self.W3(x)

# 		return outputs

# m = SinModel()
# m.compile(loss=tf.losses.MeanSquaredError(),
# 	optimizer=tf.keras.optimizers.Adam(1e-2))
# m.fit(x1, y1, epochs=30)

# # pred = tf.squeeze(m(x1)).numpy()

# # plt.plot(x,y, label='true')
# # plt.plot(x,pred,label='pred')
# # plt.legend()
# # plt.show()

# x2 = tf.range(-5,10,0.01)
# y2 = tf.math.sin(x2)

# x3 = x2[tf.newaxis,:,tf.newaxis]
# pred = tf.squeeze(m(x3)).numpy()
# plt.plot(x2,pred,label='pred')
# plt.plot(x2,y2,label='true')
# plt.legend()
# plt.show()

#--------------------------------------- model 2 ----------------------------------
x = tf.range(-5,5,0.01)
y = tf.math.sin(x)

x1 = x[:,tf.newaxis,tf.newaxis]


class SineLSTMLayer(tf.keras.layers.Layer):
	def __init__(self):
		super(SineLSTMLayer,self).__init__()
		
	def call(self, inputs):
		

m = tf.keras.models.Sequential()
m.add(tf.keras.layers.LSTM(10, stateful=True, batch_input_shape=[1,1,1]))
m.add(tf.keras.layers.Dense(1))

m.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(1e-3))
m.fit(x1,y[:,tf.newaxis], epochs=100, shuffle=False)

x2 = tf.range(-5,10,0.01)
y2 = tf.math.sin(x2)

x3 = x2[:,tf.newaxis,tf.newaxis]

pred = []
for i in range(1,len(x3)):
	pred.append(m(x3[i-1:i,:,:]).numpy())


# plt.plot(x2,y2,label='true')
# plt.plot(x2,pred,label='pred')
# plt.legend()
# plt.show()