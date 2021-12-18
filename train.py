# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:47:15 2021

@author: s1253
"""
from transformers import BertTokenizer
import tensorflow as tf
from ops import EncoderLayer, DecoderLayer, create_masks
import os 
import time
import pickle
import numpy as np



PRETRAINED_MODEL_NAME = "bert-base-chinese" 

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# loading data

checkpoint_path = "model"


'''
def load_data(file_path):
    with open(file_path, "r", encoding = 'utf-8') as file:
        
        lines = file.readlines()
        q = []
        a = []
        for i in range(len(lines)):
            if i % 3 == 0:
                q.append(lines[i].strip())
            elif i % 3 == 1:
                a.append(lines[i].strip())
            else:
                pass
               
            
    
    return q, a

q, a = load_data('chatData.txt')

q_list = []
a_list = []

for i in q:
    tokens = tokenizer.tokenize(i)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    q_list.append(ids)
    

for i in a:
    tokens = tokenizer.tokenize(i)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    a_list.append(ids)
    
with open('./q_list.pickle', 'wb') as q1:
    pickle.dump(q_list, q1)

with open('./a_list.pickle', 'wb') as a1:
    pickle.dump(a_list, a1)
'''

q_list = pickle.load(open('./data/q_list.pickle', 'rb'))
a_list = pickle.load(open('./data/a_list.pickle', 'rb'))


def filter_max_length(token_a, token_b):
  
  return tf.logical_and(tf.size(token_a) <= 30,
                        tf.size(token_b) <= 30)
def encode(en_t, zh_t):
 
  en_indices = [tokenizer.vocab_size] +list(en_t.numpy()) + [tokenizer.vocab_size + 1]
 
  zh_indices = [tokenizer.vocab_size] +list(zh_t.numpy()) + [tokenizer.vocab_size + 1]
  
  return en_indices, zh_indices

def tf_encode(en_t, zh_t):
  return tf.py_function(encode, [en_t, zh_t], [tf.int64, tf.int64])

BUFFER_SIZE = 300
BATCH_SIZE = 64
x = q_list
y = a_list

train_dataset = tf.data.Dataset.from_generator(
    lambda: iter(zip(x, y)), output_types=(tf.int64, tf.int64))

train_dataset = (train_dataset.map(tf_encode)
                 .cache()
                 .filter(filter_max_length)
                 .shuffle(BUFFER_SIZE)
                 .padded_batch(batch_size=BATCH_SIZE,padded_shapes=([-1], [-1])))

vocab_size_en = tokenizer.vocab_size + 2
vocab_size_zh = tokenizer.vocab_size + 2

#%%

'''-------------layer preprocess--------------------------------------------------------'''

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  sines = np.sin(angle_rads[:, 0::2])
  
  cosines = np.cos(angle_rads[:, 1::2])
  
  pos_encoding = np.concatenate([sines, cosines], axis=-1)
  
  pos_encoding = pos_encoding[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)



class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 rate=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
        
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, mask):
        input_seq_len = tf.shape(x)[1]
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :input_seq_len, :]
        
        
        x = self.dropout(x)
        
        for i, enc_layer in enumerate(self.enc_layers):
          x = enc_layer(x, mask)      
        
        return x 



class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
                 rate=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
  
    def call(self, x, enc_output, 
             combined_mask, inp_padding_mask):
    
        tar_seq_len = tf.shape(x)[1]
        attention_weights = {} 
        
        x = self.embedding(x)  # (batch_size, tar_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :tar_seq_len, :]
        x = self.dropout(x)
        
        
        for i, dec_layer in enumerate(self.dec_layers):
          x, block1, block2 = dec_layer(x, enc_output, 
                                        combined_mask, inp_padding_mask)
          
          attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
          attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        
        # x.shape == (batch_size, tar_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
  
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, rate)
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, rate)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
  
  
    def call(self, inp, tar, enc_padding_mask, 
             combined_mask, dec_padding_mask):

        enc_output = self.encoder(inp, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        
        
        dec_output, attention_weights = self.decoder(
            tar, enc_output,  combined_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        
        return final_output, attention_weights



#%%
# training


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask   
  
  return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


num_layers = 3
d_model = 512
dff = 2048
num_heads = 8

input_vocab_size = tokenizer.vocab_size + 2
target_vocab_size = tokenizer.vocab_size + 2
dropout_rate = 0.1  


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) * 0.2



learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

Transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)


run_id = f"{d_model}d_{num_heads}heads_{dff}dff"

checkpoint_path = os.path.join(checkpoint_path, run_id)

ckpt = tf.train.Checkpoint(Transformer,
                           optimizer=optimizer)


ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    
    last_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print(f'已讀取最新的 checkpoint，模型已訓練 {last_epoch} epochs。')
else:
    last_epoch = 0
    print("沒找到 checkpoint，從頭訓練。")




train_step_signature = [ tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64), ]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
      
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks()(inp, tar_inp)
    
    with tf.GradientTape() as tape:
      predictions, attention_weights = Transformer(inp, tar_inp,enc_padding_mask, 
                                                   combined_mask, dec_padding_mask)
                                
                                                   
                                                   
                                
      loss = loss_function(tar_real, predictions)
    
    gradients = tape.gradient(loss, Transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, Transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(tar_real, predictions)


EPOCHS =50
print(f"此超參數組合的 Transformer 已經訓練 {last_epoch} epochs。")
print(f"剩餘 epochs：{min(0, last_epoch - EPOCHS)}")




for epoch in range(last_epoch, EPOCHS):
    start = time.time()
    
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    for (step_idx, (inp, tar)) in enumerate(train_dataset):
        
        train_step(inp, tar)  
    
     
    if (epoch + 1) % 1 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                           ckpt_save_path))
    

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                  train_loss.result(), 
                                                  train_accuracy.result()))
    
    print('Time taken for 1 epoch: {} mins\n'.format((time.time() - start)/60))
  



start_token = [tokenizer.vocab_size]

def sample_from_distribution(preds):
    
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
       
    return np.random.choice(len(preds), p=preds)

    
        
    
def evaluate(inp_sentence):
  
    start_token = [tokenizer.vocab_size]
    end_token = [tokenizer.vocab_size + 1]
    a = tokenizer.encode(inp_sentence)
    a.pop(0)
    a.pop(-1)
    
    inp_sentence = start_token + a + end_token
    #print(inp_sentence)
    encoder_input = tf.expand_dims(inp_sentence, 0)
    
    
    decoder_input = [tokenizer.vocab_size]
    output = tf.expand_dims(decoder_input, 0)  
  
    for i in range(30):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks()(
            encoder_input, output)
          
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = Transformer(encoder_input, 
                                                     output,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
        
        
        # give ouput some randomness
        
        predictions = predictions[: , -1:, :]
        
        predictions = np.squeeze(predictions)
        
        predictions = sample_from_distribution(predictions)
        
        predicted_id = np.expand_dims(predictions, axis=(0, 1))
        '''
        
        predictions = predictions[: , -1:, :]  # (batch_size, 1, vocab_size)
    
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        '''
        
        if tf.equal(predicted_id, tokenizer.vocab_size + 1):
            
            return tf.squeeze(output, axis=0), attention_weights
        
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

#Transformer.summary()

#%%

# predict sentence

from opencc import OpenCC

cc = OpenCC('tw2sp')
bb = OpenCC('s2twp')




while(True):
    sentence = input('please input sentence:')
    print()
    print("me:", sentence)
    sentence = cc.convert(sentence)
    predicted_seq, _ = evaluate(sentence)
    
    target_vocab_size = tokenizer.vocab_size
    predicted_seq_without_bos_eos = [idx for idx in predicted_seq if idx < target_vocab_size]
    predicted_sentence = tokenizer.decode(predicted_seq_without_bos_eos)
    predicted_sentence = bb.convert(predicted_sentence)
    
  
    if sentence == '-1':
        break

    print("-" * 20)
    print("Alice:", predicted_sentence)






