# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 17:39:37 2022

@author: Think
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 12:02:04 2022

@author: Think

use seq2seq to simulate x+y 
"""

import tensorflow as tf 
import numpy as np 

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

# reverse data to have better results
inp_array = inp_array[:,::-1]
    
#%% split train and test
train_num = int(len(inp_array) * 0.6)
train_inp, train_targ = inp_array[:train_num], targ_array[:train_num]
test_inp, test_targ = inp_array[train_num:], targ_array[train_num:]
              
#%% create dataset 
batch_size = 10000
train_data = tf.data.Dataset.from_tensor_slices((train_inp, train_targ))\
               .batch(batch_size, drop_remainder=True)

#%% model 2: reverse + peeky, encoder is not changed, decoder is changed
class PeekyEncoder(tf.keras.Model):
    def __init__(self, embed_dim, hid_dim):
        super(PeekyEncoder, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(13,
                                                         embed_dim)
        self.lstm_layer = tf.keras.layers.LSTM(hid_dim,
                                               activation='tanh',
                                               return_sequences=False,
                                               return_state=True,
                                               stateful=False)
    def call(self, inp):
        x = self.embedding_layer(inp)
        o, h, c = self.lstm_layer(x)
        return o, h, c

class PeekyDecoder(tf.keras.Model):
    def __init__(self, embed_dim, hid_dim):
        super(PeekyDecoder, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(13,
                                                     embed_dim)
        self.lstm_layer = tf.keras.layers.LSTM(hid_dim,
                                               return_state=True,
                                               stateful=False,
                                               return_sequences=False,
                                               activation='tanh')
        self.dense_layer = tf.keras.layers.Dense(hid_dim)
        self.softmax_layer = tf.keras.layers.Dense(13)
        
    def call(self, inp, encoder_h, encoder_c, h, c):
        '''
        INPUT
            inp          shape = [batch_size, 1]
            encoder_h    shape = [batch_size, hid_dim]
            encoder_c    shape = [batch_size, hid_dim]
            h            shape = [batch_size, hid_dim], previous state of decoder
            c            shape = [batch_size, hid_dim], previous state of decoder
        '''
        x = self.embedding_layer(inp) # [batch_size, 1, embed_dim]
        x = tf.concat([x, encoder_h[:,tf.newaxis,:]], axis=-1)
        x, h, c = self.lstm_layer(x, 
                                  initial_state=(h, c))
        x = tf.concat([x, encoder_h], axis=-1) # input x = [batch_size, hid_dim]
        x = self.dense_layer(x) #[batch_size, hid_dim]
        x = self.softmax_layer(x) #[batch_size, 13]
        return x, h, c
    
#%% train model 2: setting
embed_dim = 10
hid_dim = 20
learning_rate = 1e-3
encoder = PeekyEncoder(embed_dim, hid_dim)
decoder = PeekyDecoder(embed_dim, hid_dim)

loss_func = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def train_a_batch(x,y, encoder, decoder):
    with tf.GradientTape() as tape:
        output, encoder_h, encoder_c = encoder(x)
        length = 0
        current = y[:,length:(length+1)] # [batch_size,1]
        
        # pred shape = [batch_size, char number]
        pred, hid_state, c_state = decoder(current, 
                                           encoder_h, encoder_c, 
                                           encoder_h, encoder_c)
        
        loss = tf.reduce_mean(loss_func(y[:,length+1], pred))
        while length < 3:
            length += 1
            current = y[:,length][:,tf.newaxis]
            pred, hid_state, c_state = decoder(current, 
                                               encoder_h, encoder_c,
                                               hid_state, c_state)
            loss += tf.reduce_mean(loss_func(y[:,length+1], pred))
        
        variables = encoder.trainable_variables + decoder.trainable_variables
        grad = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grad, variables))
        
    return loss       

def train_step(train_data, encoder, decoder, train_num, batch_size):
    steps = int(train_num / batch_size)
    loss= 0
    for x, y in train_data.take(steps):
        loss += train_a_batch(x,y, encoder, decoder)
    return loss / steps

def predict(x, encoder, decoder):
    out, encoder_h, encoder_c = encoder(x)
    length = 0
    
    # init pred to be the index of '_'
    pred = tf.constant([[10]]*x.shape[0], dtype=tf.int64)
    total_prediction = pred
    
    # the start state of decoder is the output of encoder
    pred_logit, h_state, c_state = decoder(pred, encoder_h, encoder_c, encoder_h, encoder_c) 
    pred = tf.argmax(pred_logit, axis=1)[:,tf.newaxis] #[num_examples,1]
    total_prediction = tf.concat([total_prediction, pred], axis=1)
    length = 1
    while length < 4:
        pred_logit, h_state, c_state = decoder(pred, encoder_h, encoder_c, h_state, c_state)
        pred = tf.argmax(pred_logit, axis=1)[:,tf.newaxis] #[num_examples,1]
        total_prediction = tf.concat([total_prediction, pred], axis=1)
        length += 1
        
    return total_prediction

def pred_accuracy_func(y_true, y_pred):
    loss = tf.pow(y_true - y_pred, 2)
    loss = tf.reduce_sum(loss, axis=1)
    accuracy = tf.reduce_sum(tf.cast(loss == 0, dtype=tf.float32)) / len(loss)
    return accuracy

#%% model 2: train
nEpochs = 10
for epoch in range(nEpochs):
    loss = train_step(train_data, encoder, decoder, train_num, batch_size)
    y_pred = predict(test_inp, encoder, decoder)
    pred_acc = pred_accuracy_func(test_targ, y_pred)
    print("epoch {}, loss {:.2f}, test accuracy {:.2f}".format(epoch, loss, pred_acc))


#%% model 1: save model
# encoder.save_weights('encoder_weights')
# decoder.save_weights('decoder_weights')
