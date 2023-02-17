import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
import pickle 

#%% model
def scaled_dot_product_attention(inputs):
	'''
	INPUT
		Q, V, K   [batch_size, num_heads, seq_len, depth]
		mask      [batch_size, num_heads, seq_len, seq_len]
	'''
	Q = inputs['query']
	K = inputs['key']
	V = inputs['value']
	mask = inputs['mask']

	# similarity between query and key
	QK = tf.matmul(Q, K, transpose_b = True) # [batch_size, num_heads, seq_len, seq_len]
	QK = QK / tf.math.sqrt(tf.cast(Q.shape[-1], tf.float32)) # scale the similarity

	if mask is not None:
		QK += (mask * (-1e9)) # remove the effect of masked values by setting it to -inf

	QK = tf.math.softmax(QK, axis=-1) # convert to probability

	attention = tf.matmul(QK, V) # [batch_size, num_heads, seq_len, depth]

	return attention 

class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()

		assert d_model % num_heads == 0

		self.d_model = d_model
		self.num_heads = num_heads
		self.depth = d_model // num_heads

		self.query_dense = tf.keras.layers.Dense(d_model)
		self.key_dense = tf.keras.layers.Dense(d_model)
		self.value_dense = tf.keras.layers.Dense(d_model)

		self.fc = tf.keras.layers.Dense(d_model)

	def split_heads(self, x):
		'''
		INPUT
			x    [batch_size, seq_len, d_model]
		'''
		batch_size = x.shape[0]
		seq_len = x.shape[1]
		x = tf.reshape(x, [batch_size, seq_len, self.num_heads, self.depth])
		x = tf.transpose(x, [0, 2, 1, 3]) # [batch_size, num_heads, seq_len, depth]

		return x 

	def call(self, inputs):
		'''
		INPUT
			inputs is a dict, having inputs.keys() = ['query', 'key', 'value', 'mask']

		'''
		Q = inputs['query']
		K = inputs['key']
		V = inputs['value']
		mask = inputs['mask']

		Q = self.query_dense(Q) # [batch_size, seq_len, d_model]
		K = self.key_dense(K)
		V = self.value_dense(V)

		Q = self.split_heads(Q)
		K = self.split_heads(K)
		V = self.split_heads(V)

		contents = scaled_dot_product_attention({
			'query': Q,
			'key': K,
			'value': V,
			'mask': mask
			})

		contents = tf.transpose(contents, [0, 2, 1, 3]) # [batch_size, seq_len, num_heads, depth]
		contents = tf.reshape(contents, [Q.shape[0], contents.shape[1], self.d_model])

		contents = self.fc(contents)

		return contents

class PositionalEncoding(tf.keras.Model):
	def __init__(self, vocab_size, d_model):
		super(PositionalEncoding, self).__init__()
		self.d_model = d_model
		self.embedding_layer = tf.keras.layers.Embedding(vocab_size, d_model)

	def positional_encoding(self, seq_len):
		'''
		OUTPUT
			[seq_len, d_model]
		'''
		depth = self.d_model // 2
		angle = tf.pow(1e4, tf.range(depth,dtype=tf.float32) * 2 / tf.cast(self.d_model, tf.float32))

		pos_enc = tf.range(seq_len, dtype=tf.float32)[:,tf.newaxis] / angle[tf.newaxis,:]

		pos_enc_1 = tf.math.sin(pos_enc)[:,tf.newaxis,:] # [seq_len, 1, depth]
		pos_enc_2 = tf.math.cos(pos_enc)[:,tf.newaxis,:] # [seq_len, 1, depth]

		pos_enc = tf.concat([pos_enc_1, pos_enc_2], axis=1) # [seq_len, 2, depth]
		pos_enc = tf.reshape(pos_enc, [seq_len, self.d_model])

		return pos_enc 

	def call(self, x):
		'''
		INPUT 
			x     [batch_size, seq_len]
		'''
		x = self.embedding_layer(x) # [batch_size, seq_len, d_model]
		x *= tf.sqrt(tf.cast(self.d_model, tf.float32))

		seq_len = x.shape[1]
		pos_enc = self.positional_encoding(seq_len)
		return x + pos_enc[tf.newaxis,:,:]

def create_paddimg_mask(x):
	'''
	INPUT
		x    [batch_size, seq_len]
	OUTPUT
		[batch_size, 1, 1, seq_len]
	'''
	mask = tf.cast(tf.equal(x, 13), tf.float32)
	return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
	'''
	INPUT
		x    [batch_size, seq_len]
	OUTPUT
		[1, 1, seq_len, seq_len]
	'''
	seq_len = x.shape[1]
	mask = 1 - tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
	return mask[tf.newaxis, tf.newaxis, :, :]

class EncoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dropout_rate):
		super(EncoderLayer, self).__init__()

		self.attention_layer = MultiHeadAttention(d_model, num_heads)
		self.dropout_layer_1 = tf.keras.layers.Dropout(dropout_rate)
		self.layer_norm_layer_1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)

		self.dense_layer = tf.keras.layers.Dense(d_model)
		self.dropout_layer_2 = tf.keras.layers.Dropout(dropout_rate)
		self.layer_norm_layer_2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)

	def call(self, inputs):
		x = inputs[0]
		padding_mask = inputs[1]
        
		attention = self.dropout_layer_1(x)
		attention = self.layer_norm_layer_1(attention)

		attention = self.attention_layer({
			'query': attention,
			'key': attention,
			'value': attention,
			'mask': padding_mask
			})
        
		attention = x + attention
        
		outputs = self.dropout_layer_2(attention)
		outputs = self.layer_norm_layer_2(outputs)
		outputs = self.dense_layer(outputs)
		outputs = outputs + attention

		return outputs

class Encoder(tf.keras.Model):
	def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout_rate):
		super(Encoder, self).__init__()
		self.num_layers = num_layers
		self.embedding_layer = PositionalEncoding(vocab_size, d_model)
		self.encoder_layer_dict = {}
		for i in range(num_layers):
			self.encoder_layer_dict[str(i)] = EncoderLayer(d_model, num_heads, dropout_rate)

	def call(self, inputs):
		'''
		INPUT
			x      [batch_size, seq_len]
			mask   [batch_size, 1, 1, seq_len], for encoder we use padding mask
		OUTPUT
			[batch_size, seq_len, d_model]
		'''
		x = inputs[0]
		padding_mask = inputs[1]

		x = self.embedding_layer(x)

		for i in range(self.num_layers):
			x = self.encoder_layer_dict[str(i)]([x, padding_mask])

		return x 

class DecoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dropout_rate):
		super(DecoderLayer, self).__init__()

		self.self_attention_layer = MultiHeadAttention(d_model, num_heads)
		self.attend_to_enc_layer = MultiHeadAttention(d_model, num_heads)
		self.dense_layer = tf.keras.layers.Dense(d_model)
		self.fc = tf.keras.layers.Dense(d_model)

		self.dropout_layer_1 = tf.keras.layers.Dropout(dropout_rate)
		self.dropout_layer_2 = tf.keras.layers.Dropout(dropout_rate)
		self.dropout_layer_3 = tf.keras.layers.Dropout(dropout_rate)

		self.layer_norm_layer_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layer_norm_layer_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layer_norm_layer_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs):
		'''
		INPUT
			x 					[batch_size, dec_seq_len, d_model]
			padding_mask		[batch_size, 1, 1, enc_seq_len]
			look_ahead_mask 	[1, 1, dec_seq_len, dec_seq_len]
			enc_context 		[batch_size, enc_seq_len, d_model]
		'''
		x, padding_mask, look_ahead_mask, enc_context = inputs 
        
		self_attention = self.dropout_layer_1(x)
		self_attention = self.layer_norm_layer_1(self_attention)
        
		self_attention = self.self_attention_layer({
			'query': self_attention,
			'key': self_attention,
			'value': self_attention,
			'mask': look_ahead_mask
			})
		self_attention = self_attention + x
		
		attention = self.dropout_layer_2(self_attention)
		attention = self.layer_norm_layer_2(attention)
		attention = self.attend_to_enc_layer({
			'query': attention,
			'key': enc_context,
			'value': enc_context,
			'mask': padding_mask
			})
		attention = attention + self_attention

		outputs = self.dropout_layer_3(attention)
		outputs = self.layer_norm_layer_3(outputs)
		outputs = self.dense_layer(outputs)
		outputs = outputs + attention

		return outputs

class Decoder(tf.keras.Model):
	def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout_rate):
		super(Decoder, self).__init__()

		self.num_layers = num_layers
		self.embedding_layer = PositionalEncoding(vocab_size, d_model)
		self.deocder_layer_dict = {}
		for i in range(num_layers):
			self.deocder_layer_dict[str(i)] = DecoderLayer(d_model, num_heads, dropout_rate)

	def call(self, inputs):
		'''
		INPUT
			x 					[batch_size, dec_seq_len]
			padding_mask		[batch_size, 1, 1, enc_seq_len]
			look_ahead_mask 	[1, 1, dec_seq_len, dec_seq_len]
			enc_context 		[batch_size, enc_seq_len, d_model]
		'''
		x, padding_mask, look_ahead_mask, enc_context = inputs
		x = self.embedding_layer(x)

		for i in range(self.num_layers):
			x = self.deocder_layer_dict[str(i)]([x, padding_mask, look_ahead_mask, enc_context])

		return x 

class Transformer(tf.keras.Model):
	def __init__(self, enc_vocab_size, dec_vocab_size, d_model, num_heads, num_layers, dropout_rate):
		super(Transformer, self).__init__()
		self.encoder = Encoder(enc_vocab_size, d_model, num_heads, num_layers, dropout_rate)
		self.decoder = Decoder(dec_vocab_size, d_model, num_heads, num_layers, dropout_rate)
		self.dense_layer = tf.keras.layers.Dense(14)

	def call(self, inputs):
		'''
		x    [batch_size, enc_seq_len]
		y    [batch_size, dec_seq_len]
		'''
		x, y = inputs
		padding_mask = create_paddimg_mask(x)
		look_ahead_mask = create_look_ahead_mask(y)

		enc_context = self.encoder([x, padding_mask])
		outputs = self.decoder([y, padding_mask, look_ahead_mask, enc_context])
		outputs = self.dense_layer(outputs)

		return outputs

def loss_func(y_true, y_pred):
	'''
	INPUT
		y_true    [batch_size, seq_len]
		y_pred    [batch_size, seq_len, vocab_size]
	'''
	loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
		reduction='none')(y_true, y_pred)
	mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

	loss *= mask 

	return tf.reduce_mean(loss)

#%% training settings
def predict(model, x):
	pred = tf.constant([[11]] * x.shape[0], dtype=tf.int32)
	total_prediction = pred 
	for pos in range(4):
		pred = model([x,total_prediction])
		pred = pred[:,-1,:]
		pred = tf.cast(tf.argmax(pred,axis=-1), tf.int32)
		total_prediction = tf.concat([total_prediction, pred[:,tf.newaxis]], axis=-1)
	return total_prediction

def test_accuracy(y_true, y_pred):
	diff = tf.reduce_sum(tf.pow(y_true - y_pred, 2), axis=1)
	diff = tf.cast(tf.equal(diff, 0), tf.float32)
	return tf.reduce_mean(diff)

#%% generate data
inp = list()
targ = list()
for i in range(1000):
    for j in range(1000):
        r = str(i) + '+' + str(j)
        if len(r) < 7:
            r = r + ' ' * (7-len(r))
        inp.append(r)
        
        r = '_' + str(i+j)
        if len(r) < 5:
            r = r + ' ' * (5 - len(r))
        targ.append(r)
        
#%% prepare data for encoding: str to nums
num2str_dict = dict(list(enumerate([str(i) for i in range(10)] + ['_', '+', ' '])))
str2num_dict = {v:k for (k,v) in num2str_dict.items()}

inp_array = list()
for i in inp:
    inp_array.append([str2num_dict[j] for j in i])
    
targ_array = list()
for i in targ:
    targ_array.append([str2num_dict[j] for j in i])

#%% change data to np.array
permu_idx = np.random.permutation(len(inp_array))
inp_array = np.array(inp_array)[permu_idx]
targ_array = np.array(targ_array)[permu_idx]

# we don't want to use 0 padding 
inp_array = inp_array + 1
targ_array = targ_array + 1
    
#%% split train and test
train_num = int(len(inp_array) * 0.6)
train_inp, train_targ = inp_array[:train_num], targ_array[:train_num]
test_inp, test_targ = inp_array[train_num:], targ_array[train_num:]

#%% change to tensorflow dataset
batch_size = 64
train_data = tf.data.Dataset.from_tensor_slices((train_inp, train_targ))\
               .batch(batch_size, drop_remainder=True)

#%% custom schedule
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

#%% train the model 
enc_vocab_size = 14
dec_vocab_size = 14
d_model = 64
num_heads = 8
num_layers = 2
dropout_rate = 0.1
#learning_rate = CustomSchedule(d_model)
learning_rate = 1e-3
batch_size = 64
epochs = 50
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = 0.9, beta_2 = 0.98, epsilon=1e-9)
model = Transformer(enc_vocab_size, dec_vocab_size, d_model, num_heads, num_layers, dropout_rate)

def train_a_batch(x,y):
    with tf.GradientTape() as tape:
        pred = model([x,y])
        loss = loss_func(y[:,1:], pred[:,:-1,:])
        variables = model.trainable_variables
        gradient = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradient, variables))
    return loss 

steps = train_num // batch_size
# steps = 1

loss_list = []
test_acc_list = []
for epoch in range(epochs):
    loss = 0
    for x,y in train_data.take(steps):
        loss += train_a_batch(x,y)
    loss /= steps
    loss_list.append(loss)

    test_pred = predict(model, test_inp)
    test_acc = test_accuracy(test_targ, test_pred)
    test_acc_list.append(test_acc)
    print('epoch {}, loss {:.2f}, test accuracy {:.2f}'.format(epoch, loss, test_acc))

with open('loss_list.pkl', 'wb') as f:
    pickle.dump(loss_list, f)
    
with open('test_acc.pkl', 'wb') as f:
    pickle.dump(test_acc_list, f)
    
model.save_weights('transformer')