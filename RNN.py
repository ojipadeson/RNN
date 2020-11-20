# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:52:52 2020

@author: MI
"""

import tensorflow_datasets as tfds
import tensorflow as tf

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# gpus = tf.config.list_logical_devices()
gpus= tf.config.experimental.list_physical_devices('GPU') # tf2.1版本该函数不再是experimental
print(gpus) # 前面限定了只使用GPU1(索引是从0开始的,本机有2张RTX2080显卡)
tf.config.experimental.set_memory_growth(gpus[0], True)
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]
# )
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

# import matplotlib.pyplot as plt

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# def plot_graphs(history, metric):
#     plt.plot(history.history[metric])
#     plt.plot(history.history['val_'+metric], '')
#     plt.xlabel("Epochs")
#     plt.ylabel(metric)
#     plt.legend([metric, 'val_'+metric])
#     plt.show()
  
dataset, info = tfds.load('imdb_reviews/subwords8k',
                          # data_dir = '/root/Desktop',
                          # data_dir = 'C:/Users/MI/tensorflow_datasets/imdb_reviews/subwords8k',
                          with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

print('Vocabulary size: {}'.format(encoder.vocab_size))

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))

assert original_string == sample_string

for index in encoded_string:
    print('{} ----&gt; {}'.format(index, encoder.decode([index])))
    
BUFFER_SIZE = 100
BATCH_SIZE = 32

padded_shapes = ([None],())
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE,padded_shapes)

test_dataset = test_dataset.padded_batch(BATCH_SIZE,padded_shapes)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset, 
                    validation_steps=10)

test_loss, test_acc = model.evaluate(test_dataset)

model.save('rnn_film.h5')
model = tf.keras.models.load_model('rnn_film.h5')

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sample_pred_text, pad):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)
    
    if pad:
      encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
    
    return (predictions)

# predict on a sample text without padding.

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)

# plot_graphs(history, 'accuracy')
# plot_graphs(history, 'loss')
