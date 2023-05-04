import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LayerNormalization
from tensorflow.keras.layers import Conv1D, UpSampling1D, AveragePooling1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean
import numpy as np


class Generator(Model):
    def __init__(self, latent_dim, num_heads, head_size, num_transformer_blocks):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_transformer_blocks = num_transformer_blocks

        self.dense1 = Dense(units=32, activation='relu')
        self.dense2 = Dense(units=64, activation='relu')
        self.dense3 = Dense(units=128, activation='relu')
        self.dense4 = Dense(units=256, activation='relu')
        self.dense5 = Dense(units=512, activation='relu')
        self.dense6 = Dense(units=1024, activation='relu')
        self.dense7 = Dense(units=2048, activation='relu')
        self.reshape = Reshape(target_shape=(2048, 1))

        self.transformer_blocks = []
        for i in range(self.num_transformer_blocks):
            transformer_block = TransformerBlock(num_heads=self.num_heads, head_size=self.head_size)
            self.transformer_blocks.append(transformer_block)

        self.flatten = Flatten()
        self.dense8 = Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.reshape(x)

        for i in range(self.num_transformer_blocks):
            x = self.transformer_blocks[i](x)

        x = self.flatten(x)
        output = self.dense8(x)

        return output


class TransformerBlock(Model):
    def __init__(self, num_heads, head_size):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        self.layer_norm1 = LayerNormalization()
        self.multihead_attention = MultiHeadAttention(num_heads=self.num_heads, head_size=self.head_size)
        self.dropout1 = Dropout(rate=0.1)
        self.layer_norm2 = LayerNormalization()
        self.dense1 = Dense(units=2048, activation='relu')
        self.dropout2 = Dropout(rate=0.1)
        self.dense2 = Dense(units=2048, activation='relu')
        self.dropout3 = Dropout(rate=0.1)

    def call(self, inputs):
        x = self.layer_norm1(inputs)
        x = self.multihead_attention(x)
        x = self.dropout1(x)
        x = inputs + x

        x = self.layer_norm2(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.dropout3(x)
        x = inputs + x

        return x


class MultiHeadAttention(Model):
    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.query_dense = Dense(units=self.num_heads * self.head_size, activation=None)
        self.key_dense = Dense(units=self.num_heads * self.head_size, activation=None)
        self.value_dense = Dense(units=self.num_heads * self.head_size, activation=None)
        self.output_dense = Dense(units=self.num_heads * self.head_size, activation=None)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Linear projections
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Reshape to num_heads x head_size
        query = tf.reshape(query, shape=(batch_size, -1, self.num_heads, self.head_size))
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.reshape(key, shape=(batch_size, -1, self.num_heads, self.head_size))
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.reshape(value, shape=(batch_size, -1, self.num_heads, self.head_size))
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        # Scaled dot-product attention
        attention_logits = tf.matmul(query, key, transpose_b=True)
        attention_logits = tf.math.divide(attention_logits, tf.math.sqrt(tf.cast(self.head_size, dtype=tf.float32)))
        attention_scores = tf.nn.softmax(attention_logits, axis=-1)
        attention_output = tf.matmul(attention_scores, value)

        # Reshape back to input shape
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, shape=(batch_size, -1, self.num_heads * self.head_size))

        # Linear projection
        output = self.output_dense(attention_output)

        return output

