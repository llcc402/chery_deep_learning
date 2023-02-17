#%% 
import tensorflow as tf

#---------------------------- train data -------------------------------
#%% read data
with open('E:\datasets\qi18naacl-dataset\datasets\pt_to_en\en.train', 'r', encoding='utf8') as f:
    en = f.readlines()
    
with open('E:\datasets\qi18naacl-dataset\datasets\pt_to_en\pt.train', 'r', encoding='utf8') as f:
    pt = f.readlines()
    
print('English version has {} sentences'.format(len(en)))
print('Portuguese version has {} sentences'.format(len(pt)))
print('\n')
print('The first sentence of the English version is:\n{}'.format(en[0]))
print('The first sentence of the Portuguese version is:\n{}'.format(pt[0]))

#%% add start and end token
def add_start_end(x):
    '''
    INPUT
        x is a string
    '''
    return '<START> ' + x + ' <END>'

en = [add_start_end(x) for x in en]
pt = [add_start_end(x) for x in pt]

#%% tokenize text
tokenizer_en = tf.keras.preprocessing.text.Tokenizer()
tokenizer_en.fit_on_texts(en)

tokenizer_pt = tf.keras.preprocessing.text.Tokenizer()
tokenizer_pt.fit_on_texts(pt)

# word token to integers
en_seq = tokenizer_en.texts_to_sequences(en)
pt_seq = tokenizer_pt.texts_to_sequences(pt)

# pad sequences to have the same length
en_seq_padded = tf.keras.preprocessing.sequence.pad_sequences(en_seq, padding='post')
pt_seq_padded = tf.keras.preprocessing.sequence.pad_sequences(pt_seq, padding='post')

#%% build dataset
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((en_seq_padded, pt_seq_padded)).shuffle(51785).batch(batch_size)

#------------------------------ dev data ---------------------------------
#%% read dev data
with open('E:\datasets\qi18naacl-dataset\datasets\pt_to_en\en.dev', 'r', encoding='utf8') as f:
    en_dev = f.readlines()
    
with open('E:\datasets\qi18naacl-dataset\datasets\pt_to_en\pt.dev', 'r', encoding='utf8') as f:
    pt_dev = f.readlines()
    
print('English version has {} sentences'.format(len(en_dev)))
print('Portuguese version has {} sentences'.format(len(pt_dev)))
print('\n')
print('The first sentence of the English version is:\n{}'.format(en_dev[0]))
print('The first sentence of the Portuguese version is:\n{}'.format(pt_dev[0]))

#%% add start and end token
en_dev = [add_start_end(x) for x in en_dev]
pt_dev = [add_start_end(x) for x in pt_dev]

#%% tokenize
en_dev_seq = tokenizer_en.texts_to_sequences(en_dev)
pt_dev_seq = tokenizer_pt.texts_to_sequences(pt_dev)

#%% padding
en_dev_seq_padded = tf.keras.preprocessing.sequence.pad_sequences(en_dev_seq, padding='post')
pt_dev_seq_padded = tf.keras.preprocessing.sequence.pad_sequences(pt_dev_seq, padding='post')

#%%
dev_dataset = tf.data.Dataset.from_tensor_slices((en_dev_seq_padded, pt_dev_seq_padded)).batch(batch_size)

#-------------------------- modelling ------------------------------
def scaled_dotted_product_attention(Q, K, V, mask=None):
    '''
    INPUT
        Q      [batch_size, num_heads, seq_length, depth]
        K      [batch_size, num_heads, seq_length, depth]
        V      [batch_size, num_heads, seq_length, depth]
        mask   [batch_size, num_heads, seq_len_1, seq_len_2]
               for padding mask, we can set its shape [batch_size, 1, 1, key_seq_len]
                 - reason: for all heads and all query words in the same sentence, the padded words are the same,
                   recall that the dot product of (QK^T)_ij is the similarity between query word i and key word j,
                   and i should not attend to j if j is a padding 0
               for look ahead mask, we can set its shape = [batch_size, 1, query_seq_len, query_seq_len]
                 - reason: for self attention in the decoder, word i should not attend to the word j before j has 
                   been disclosed. Hence, the matrix of the look ahead mask in the last 2 dims should be a lower
                   triangluar matrix with diagonal values > 0, which denotes the similarities between i and itself
        depth  the dimension of each head, should be d_model / head_num 
    OUTPUT
        context    [batch_size, num_heads, seq_length, depth]
                   the output is a new representation of the words with attention information stored in the new
                   representation
    '''
    QK = tf.matmul(Q, K, transpose_b=True) # [batch_size, num_heads, seq_length, seq_length]
    QK = QK / tf.math.sqrt(tf.cast(Q.shape[-1], tf.float32)) # [batch_size, num_heads, seq_length, seq_length]
    
    # mask out future information
    if mask is not None:
        QK += (mask * -1e9) # make the similarity really small if the query word is padding 0 or look ahead 0
        
    attention_weights = tf.nn.softmax(QK, axis=-1) # [batch_size, num_heads, seq_length, seq_length]
    
    outputs = tf.matmul(attention_weights, V)
    return outputs # [batch_size, num_heads, seq_length, depth]
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        '''
        INPUT
            d_model      dimension of the encoded representations, not the embedding dim  
            num_heads    how many different spaces for self attention
        '''
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.query_dense = tf.keras.layers.Dense(self.d_model)
        self.key_dense = tf.keras.layers.Dense(self.d_model)
        self.value_dense = tf.keras.layers.Dense(self.d_model)
        
        self.fc = tf.keras.layers.Dense(self.d_model)
        
    def split_heads(self, X):
        '''
        INPUT
            X     a tensor, [batch_size, seq_length, d_model]
        OUTPUT
            X     [batch_size, num_heads, seq_length, depth]
                  d_model = num_heads * depth
        '''
        batch_size, seq_length = X.shape[0], X.shape[1]
        X = tf.reshape(X, [batch_size, seq_length, self.num_heads, self.d_model//self.num_heads])
        return tf.transpose(X, [0,2,1,3])
        
    def call(self, inputs):
        '''
        INPUT
            inputs is a dictionary so we can use model.compile
        OUTPUT
            context    [batch_size, seq_len, d_model]
        '''
        Q, K, V, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size, seq_len = Q.shape[0], Q.shape[1]
        
        Q = self.query_dense(Q) # [batch_size, seq_length, d_model]
        K = self.key_dense(K)
        V = self.value_dense(V)
        
        Q = self.split_heads(Q) # [batch_size, num_heads, seq_length, depth]
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        context = scaled_dotted_product_attention(Q, K, V, mask) # [batch_size, num_heads, seq_length, depth]
        
        # concat all heads together
        context = tf.transpose(context, [0,2,1,3]) # [batch_size, seq_length, num_heads, depth]
        context = tf.reshape(context, shape = [batch_size, seq_len, self.d_model]) # [batch_size, seq_length, d_model]
        
        context = self.fc(context)
        
        return context 
    
# use tensorflow funtional api to have compact code
class EncoderSublayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(EncoderSublayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        self.attention_layer = MultiHeadAttention(d_model, num_heads)
        self.dropout_layer_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_layer_2 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm_layer_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_layer_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense_layer = tf.keras.layers.Dense(d_model)
        
    def call(self, inputs):
        '''
        INPUTS
            x     [batch_size, seq_len, d_model]
            mask  for encoder, we only use padding mask, which has dimension
                  [batch_size, 1, 1, key_seq_len], broadcast for the same head and query word
        '''
        x = inputs[0]
        mask = inputs[1]
        
        x = self.attention_layer({
            'key': x,
            'query': x,
            'value': x,
            'mask': mask
        })
        
        x = self.dropout_layer_1(x)
        x = self.layer_norm_layer_1(x + inputs[0])
        
        outputs = self.dense_layer(x)
        outputs = self.dropout_layer_2(outputs)
        outputs = self.layer_norm_layer_2(outputs + x)
        
        return outputs

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, d_model)
        
    def positional_encoding(self, pos):
        '''
        INPUT
            pos         should be seq_len
            d_embed     integer, embedding dimension
        '''
        d_embed = self.d_model
        assert d_embed % 2 == 0
        angle = tf.range(d_embed // 2, dtype=tf.float32)
        angle = tf.range(pos, dtype = tf.float32)[:,tf.newaxis] / tf.pow(1e4, 2 * angle / d_embed)[tf.newaxis,:]
        encoded_0 = tf.math.sin(angle)[:,:,tf.newaxis] #[seq_len, d_embed // 2, 1]
        encoded_1 = tf.math.cos(angle)[:,:,tf.newaxis] 

        encoded = tf.concat([encoded_0, encoded_1], axis=-1) # [seq_len, d_embed // 2, 2]
        encoded = tf.reshape(encoded, shape = [-1,d_embed])  # [seq_len, d_embed]
        return encoded
    
    def call(self, inputs):
        '''
        INPUT
            inputs     a tensor, token index of sentences, [batch_size, seq_len]
        '''
        pos = inputs.shape[1]
        x = self.embedding_layer(inputs) #[batch_size, seq_len, d_model]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # scale embedding to balance positional encoding
        
        return x + self.positional_encoding(pos)[tf.newaxis,:,:]

class EncoderModel(tf.keras.Model):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, dropout_rate):
        super(EncoderModel, self).__init__()
        self.num_layers = num_layers
        
        self.encoder_layer_dict = {}
        for i in range(num_layers):
            self.encoder_layer_dict[i] = EncoderSublayer(d_model, num_heads, dropout_rate)
            
        self.positional_enc_layer = PositionalEncodingLayer(vocab_size, d_model)
        
    def call(self, inputs):
        '''
        INPUTS
            x     [batch_size, seq_len, d_model]
            mask  for encoder, we only use padding mask, which has dimension
                  [batch_size, 1, 1, key_seq_len], broadcast for the same head and query word
        '''
        x = inputs[0]
        mask = inputs[1]
        
        x = self.positional_enc_layer(x)
        
        for _ in range(self.num_layers):
            x = self.encoder_layer_dict[i]([x, mask])
            
        return x
    
class DecoderSublayer(tf.keras.Model):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(DecoderSublayer, self).__init__()
        
        self.attention_layer_1 = MultiHeadAttention(d_model, num_heads)
        self.attention_layer_2 = MultiHeadAttention(d_model, num_heads)
        self.layer_norm_layer_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_layer_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_layer_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense_layer_1 = tf.keras.layers.Dense(d_model, activation='relu')
        self.dense_layer_2 = tf.keras.layers.Dense(d_model)
        self.dropout_layer_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_layer_2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_layer_3 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs):
        '''
        INPUT
            inputs     [x, padding_mask, future_mask, encoded_context]
                       - x is the input of the previous prediction
                       - recall keys are from encoded information, so padding_mask should have shape
                         [batch_size, 1, 1, enc_seq_len]
                       - Future mask is used prevent decoder see future unpredicted words
                         [batch_size, 1, 1, key_seq_len]
                       - encoded_context = [batch_size, seq_len, d_model]
        '''
        x, padding_mask, future_mask, encoded_context = inputs
        x1 = self.attention_layer_1({
            'key': x,
            'query': x,
            'value': x,
            'mask': future_mask
        })
        x1 = self.dropout_layer_1(x1)
        x1 = self.layer_norm_layer_1(x1 + x)
        
        x2 = self.attention_layer_2({
            'key': encoded_context,
            'query': x1,
            'value': encoded_context,
            'mask': padding_mask
        }) # masks for encoded keys
        x2 = self.dropout_layer_2(x2)
        x2 = self.layer_norm_layer_2(x2 + x1)
        
        x3 = self.dense_layer_1(x2)
        x3 = self.dense_layer_2(x3)
        x3 = self.dropout_layer_3(x3)
        x3 = self.layer_norm_layer_3(x3 + x2)
        
        return x3
    
class DecoderModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout_rate):
        super(DecoderModel, self).__init__()
        self.num_layers = num_layers
        
        self.decoder_layer_dict = {}
        for i in range(num_layers):
            self.decoder_layer_dict[i] = DecoderSublayer(d_model, num_heads, dropout_rate)
            
        self.positional_enc_layer = PositionalEncodingLayer(vocab_size, d_model)
        
    def call(self, inputs):
        '''
        INPUT
            inputs     [x, padding_mask, future_mask, encoded_context]
                       - x is the input of the previous prediction
                       - Current multi head attention layer requires 
                                 encoder and decoder 
                         to have the same [batch_size, seq_len]
                       - Future mask is used prevent decoder see future unpredicted words
                       - encoded_context = [batch_size, seq_len, d_model]
        '''
        x, padding_mask, future_mask, encoded_context = inputs
        x = self.positional_enc_layer(x)
        for _ in range(self.num_layers):
            x = self.decoder_layer_dict[i]([x, padding_mask, future_mask, encoded_context])
        return x
    
class Transformer(tf.keras.Model):
    def __init__(self,vocab_size_1, vocab_size_2, 
                 num_layers, d_model, num_heads, dropout_rate):
        super(Transformer, self).__init__()
        self.encoder = EncoderModel(vocab_size_1, num_layers, d_model, num_heads, dropout_rate)
        self.decoder = DecoderModel(vocab_size_2, d_model, num_heads, num_layers, dropout_rate)
        self.dense_layer = tf.keras.layers.Dense(vocab_size_2)
        
    # define masks
    def add_masks(self, inputs):
        x = inputs[0]
        y = inputs[1]
        padding_mask = tf.cast(tf.equal(x,0),tf.float32)
        padding_mask = padding_mask[:,tf.newaxis, tf.newaxis,:]
        future_mask = 1.0 - tf.linalg.band_part(tf.ones([y.shape[1],y.shape[1]]), -1,0)
        future_mask = future_mask[tf.newaxis,tf.newaxis,:,:]
        return padding_mask, future_mask

    def call(self, inputs):
        '''
        INPUT 
            padding_mask         [batch_size, 1, 1, enc_seq_len]
            future_mask          [batch_size, 1, dec_seq_len, dec_seq_len]
        OUTPUT
            [batch_size, dec_seq_len, vocab_size_2] the prediction logits for each word of each sentence 
            of the decoder
        '''
        x = inputs[0]
        y = inputs[1]
        padding_mask, future_mask = self.add_masks(inputs)
        enc_context = self.encoder([x, padding_mask])
        outputs = self.decoder([y, padding_mask, future_mask, enc_context])
        outputs = self.dense_layer(outputs)
        return outputs

#%% training settings
def loss_func(y_true, y_pred):
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                   reduction='none')(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true,0), tf.float32)
    loss = loss * mask
    return tf.reduce_mean(loss)

#------------------------------ English to Portugal ------------------------------
vocab_size_1 = len(tokenizer_en.word_counts)+1
vocab_size_2 = len(tokenizer_pt.word_counts)+1
num_layers = 3
d_model = 64
num_heads = 2
dropout_rate = 0.1
learning_rate = 1e-4

model = Transformer(vocab_size_1, vocab_size_2, 
                 num_layers, d_model, num_heads, dropout_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate)

def train_a_batch(x,y):
    with tf.GradientTape() as tape:
        pred = model([x,y])
        loss = loss_func(y[:,1:], pred[:,:-1,:])
        variables = model.trainable_variables
        gradient = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradient, variables))
    return loss, gradient

dev_x, dev_y = dev_dataset.as_numpy_iterator().next()

for step in range(10):
    i = 0
    for x,y in train_dataset.take(500):
        loss, grad = train_a_batch(x,y)
        dev_pred = model([dev_x, dev_y])
        dev_loss = loss_func(dev_y[:,1:], dev_pred[:,:-1,:])
        print('batch {}, train loss {:.2f}, dev loss {:.2f}'.format(i, loss, dev_loss))
        i += 1

#---------------------------------- toy data ------------------------------------
x = tf.constant([[0,1,2],[1,2,0]], dtype=tf.int32)
y = tf.constant([[1,3,3,1],[1,1,3,2]],dtype=tf.int32)

vocab_size_1 = 3
vocab_size_2 = 4
num_layers = 2
d_model = 8
num_heads = 2
dropout_rate = 0.1
learning_rate = 1e-3

model = Transformer(vocab_size_1, vocab_size_2, 
                 num_layers, d_model, num_heads, dropout_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate)

def train_a_batch(x,y):
    with tf.GradientTape() as tape:
        pred = model([x,y])
        loss = loss_func(y[:,1:], pred[:,:-1,:])
        variables = model.trainable_variables
        gradient = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradient, variables))
    return loss, gradient

for i in range(300):
    loss, grad = train_a_batch(x,y)
    print('batch {}, train loss {:.2f}, '.format(i, loss))