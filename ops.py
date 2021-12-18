# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:22:51 2021

@author: s1253
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, LayerNormalization, Dropout

# the functions are from https://www.tensorflow.org/text/tutorials/transformer

def scaled_dot_product_attention(q, k, v, mask):
    
    matmul_qk = tf.matmul(q, k, transpose_b=True)  
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32) 
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) 
    
    if mask is not None:
      scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  
    
    output = tf.matmul(attention_weights, v)  
    
    return output, attention_weights

class Maltihead_attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_head):
        super(Maltihead_attention, self).__init__()
        
        self.d_model = d_model
        self.num_head = num_head
        
        self.d_heads = int(d_model / num_head)
        
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        
        self.dense = Dense(d_model)
    
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads
        })
        return config
    
    def split_heads(self, arr, batch_size):
               
        arr = tf.reshape(arr, [batch_size, -1, self.num_head, self.d_heads])
        arr = tf.transpose(arr, perm=[0, 2, 1, 3])
        
        return arr
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(v)[0]
        
        v = self.wv(v)
        q = self.wq(q)
        k = self.wk(k)
        
        v = self.split_heads(v, batch_size)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        
        output, attention_weight = scaled_dot_product_attention(q, k, v, mask)
        
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, -1,self.d_model])
        
        output = self.dense(output)
        
        return output, attention_weight


def point_wise_feed_forward_network(d_model, dff):
  
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
      super(EncoderLayer, self).__init__()

      self.d_model = d_model
      self.num_heads = num_heads
      self.dff = dff
      
      self.mha = Maltihead_attention(d_model, num_heads)
      self.ffn = point_wise_feed_forward_network(d_model, dff)
    
      self.layernorm1 = LayerNormalization(epsilon=1e-6)
      self.layernorm2 = LayerNormalization(epsilon=1e-6)
    
      self.dropout1 = Dropout(rate)
      self.dropout2 = Dropout(rate)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff":self.dff
        })
        return config
  
    def call(self, x, mask):
      #name = name
      attn_output, attn = self.mha(x, x, x, mask)   # (batch_size, input_seq_len, d_model)
      attn_output = self.dropout1(attn_output) 
      out1 = self.layernorm1(x + attn_output)   # (batch_size, input_seq_len, d_model)
      
      ffn_output = self.ffn(out1) 
      ffn_output = self.dropout2(ffn_output)  
      out2 = self.layernorm2(out1 + ffn_output)
      
      return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
      super(DecoderLayer, self).__init__()
      
      self.d_model = d_model
      self.num_heads = num_heads
      self.dff = dff
    
      self.mha1 = Maltihead_attention(d_model, num_heads)
      self.mha2 = Maltihead_attention(d_model, num_heads)
      self.ffn = point_wise_feed_forward_network(d_model, dff)
     
      self.layernorm1 = LayerNormalization(epsilon=1e-6)
      self.layernorm2 = LayerNormalization(epsilon=1e-6)
      self.layernorm3 = LayerNormalization(epsilon=1e-6)
      
      self.dropout1 = Dropout(rate)
      self.dropout2 = Dropout(rate)
      self.dropout3 = Dropout(rate)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff":self.dff
        })
        return config
  
    
    def call(self, x, enc_output,
             combined_mask, inp_padding_mask):
      
      attn1, attn_weights_block1 = self.mha1(x, x, x, combined_mask)
      attn1 = self.dropout1(attn1)
      out1 = self.layernorm1(attn1 + x)
      
      attn2, attn_weights_block2 = self.mha2(
          enc_output, enc_output, out1, inp_padding_mask)  
      attn2 = self.dropout2(attn2)
      out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
      
      ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    
      ffn_output = self.dropout3(ffn_output)
      out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
      
      return out3, attn_weights_block1, attn_weights_block2


def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  sines = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  cosines = np.cos(angle_rads[:, 1::2])
  
  pos_encoding = np.concatenate([sines, cosines], axis=-1)
  
  pos_encoding = pos_encoding[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)


class preprocess(tf.keras.layers.Layer): 
    def __init__(self):
        super(preprocess, self).__init__()
        
    def call(self, x, input_seq_len, d_model):
        input_vocab_size = 35
        pos_encoding = positional_encoding(input_vocab_size, d_model)
        x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        x += pos_encoding[:, :input_seq_len, :]
        
        return x


def create_padding_mask(seq):
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

class create_masks(tf.keras.layers.Layer):
    def __init__(self):
        super(create_masks, self).__init__()
    
    
    def call(self, inp, tar):
        enc_padding_mask = create_padding_mask(inp)
        
        dec_padding_mask = create_padding_mask(inp)
        
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return enc_padding_mask, combined_mask, dec_padding_mask
    
    
    
    

