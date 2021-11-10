# modified_v1.py
# removes reducers
# pretrain on food101 dataset

import numpy as np
import os
import random

# tensorflow libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Input, concatenate, GlobalAveragePooling2D, Rescaling, Activation, Add, Resizing
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.models import Model
from keras.losses import SparseCategoricalCrossentropy

import time

dir = '/home/UG/jwoon006/CZ4042-THFOOD50/THFOOD50-v1'

SEED = 50
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

num_classes = 101
batch_size = 64
lr = 0.1
weight_decay = 0.0005
epochs = 200
momentum= 0.9

pretrain_train, pretrain_val = tfds.load('food101', split=['train', 'validation'], shuffle_files=True, batch_size=batch_size, as_supervised=True)

class_names = [x[0].split('/')[-1] for x in os.walk(dir+'/train')]
class_names = class_names[1:]

train_ds = image_dataset_from_directory(
    dir+'/train',
    labels='inferred',
    label_mode='int', 
    class_names=class_names,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(224,224),
)
val_ds = image_dataset_from_directory(
    dir+'/val',
    labels='inferred',
    label_mode='int', 
    class_names=class_names,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(224,224),
)

test_ds = image_dataset_from_directory(
    dir+'/test',
    labels='inferred',
    label_mode='int', 
    class_names=class_names,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(224,224)
)

resizing_layer = Resizing(224, 224)
norm_layer = Rescaling(1./255)

pretrain_train = pretrain_train.map(lambda x, y: (resizing_layer(x), y))
pretrain_val = pretrain_val.map(lambda x, y: (resizing_layer(x), y))
pretrain_train = pretrain_train.map(lambda x, y: (norm_layer(x), y))
pretrain_val = pretrain_val.map(lambda x, y: (norm_layer(x), y))

train_ds = train_ds.map(lambda x, y: (norm_layer(x), y))
val_ds = val_ds.map(lambda x, y: (norm_layer(x), y))
test_ds = test_ds.map(lambda x, y: (norm_layer(x), y))

def conv_bn_relu(num_filters, filter_size, stride, padding, name, layer_in):
  conv = Conv2D(num_filters, filter_size, stride, padding=padding, name=name, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(weight_decay),)(layer_in)
  conv_bn = BatchNormalization()(conv)
  layer_out = Activation('relu')(conv_bn)

  return layer_out

def inception_1(channel1, depth, layer_in):
  layer_out = conv_bn_relu(channel1, 1, 1, 'valid', f'nu_inception_{depth}_1x1', layer_in)

  return layer_out

def inception_2(channel2, depth, layer_in):
  layer_out = conv_bn_relu(channel2, 3, 1, 'same', f'nu_inception_{depth}_3x3', layer_in)

  return layer_out

def inception_3(channel2, channel3, depth, layer_in):
  conv2 = conv_bn_relu(channel2, 3, 1, 'same', f'nu_inception_{depth}_3x3_1', layer_in)
  layer_out = conv_bn_relu(channel3, 3, 1, 'same', f'nu_inception_{depth}_3x3_2', conv2)

  return layer_out

def inception_4(channel2, channel3, channel4, depth, layer_in):
  conv2 = conv_bn_relu(channel2, 3, 1, 'same', f'nu_inception_{depth}_3x3_1_3', layer_in)
  conv3 = conv_bn_relu(channel3, 3, 1, 'same', f'nu_inception_{depth}_3x3_2_3', conv2)
  layer_out = conv_bn_relu(channel4, 3, 1, 'same', f'nu_inception_{depth}_3x3_3_3', conv3)

  return layer_out

def nu_inception(params1, params2, params3, params4, direct_channel, depth, layer_in):
  out1 = inception_1(params1[0], depth, layer_in)
  out2 = inception_2(params2[0], depth, layer_in)
  out3 = inception_3(params3[0], params3[1], depth, layer_in)
  out4 = inception_4(params4[0], params4[1], params4[2], depth, layer_in)

  concat_layer = concatenate([out1, out2, out3, out4], axis=3)

  layer_out = conv_bn_relu(direct_channel, 1, 1, 'same', f'conv_direct_{depth}', concat_layer)

  return layer_out

def resnet_block(params1, params2, params3, params4, direct_channel, bypass_channel, depth, layer_in):
  pool = MaxPooling2D(3, 2, name=f'pool{depth}')(layer_in)
  conv1 = nu_inception(params1, params2, params3, params4, direct_channel, depth, pool)
  conv2 = conv_bn_relu(bypass_channel, 1, 1, 'valid', f'conv_bypass_{depth}', pool)

  residual = Add(name=f'residual{depth}')([conv1, conv2])
  layer_out = Activation('relu')(residual)

  return layer_out

module1 = [[16], [32], [8, 8], [8, 8, 8]]
module2 = [[32], [64], [16, 16], [16, 16, 16]]
module3 = [[64], [128], [32, 32], [32, 32, 32]]
module4 = [[128], [256], [64, 64], [64, 64, 64]]
modules = [module1, module2, module3, module4]

direct_channels = [64, 128, 256, 512]
bypass_channels = [64, 128, 256, 512]

def create_model(fine_tuning = False):
  input = Input(shape=(224, 224, 3))

  layer_out = conv_bn_relu(64, 5, 2, 'same', 'conv1', input)

  for i, module in enumerate(modules):
    layer_out = resnet_block(module[0], module[1], module[2], module[3], direct_channels[i], bypass_channels[i], i+1, layer_out)

  avg_pool = GlobalAveragePooling2D(name='avg_pool')(layer_out)
  
  if fine_tuning:
    layer_out = Dense(num_classes, name='dense_finetune')(avg_pool)
  else:
    layer_out = Dense(num_classes, name='dense')(avg_pool)

  model = Model(inputs=input, outputs=layer_out) 

  model.compile(
      optimizer= tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
  )

  model.summary()

  return model

AUTOTUNE = tf.data.AUTOTUNE

pretrain_train = pretrain_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
pretrain_val = pretrain_val.cache().prefetch(buffer_size=AUTOTUNE)

model = create_model()

checkpoint_path = "modified_v1_pretrained/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss', 
                                                 mode='min',
                                                 verbose=1)

def scheduler(epoch, lr):
  rates = [0.1, 0.01,  0.001, 0.0001]
  return tf.dtypes.cast(rates[epoch//50], tf.float32) 

sh_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# pre-train on food101 dataset
start_time = time.time()
history = model.fit(pretrain_train, epochs=epochs, validation_data=pretrain_val, callbacks=[cp_callback, sh_callback], verbose=2)
time_taken = time.time() - start_time
print("Total time taken to pre-train in seconds:", time_taken)

model = create_model(fine_tuning=False)
model.load_weights(checkpoint_path) #load all weights
x = model.layers[-2].output #remove 101 prediction layer
num_classes=50
predictions = Dense(num_classes, name='dense_finetune')(x)
model = Model(inputs=model.input, outputs=predictions)

lr = 0.0001
epochs = 100

model.compile(
      optimizer= tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
  )

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# fine-tune on THFOOD50 dataset
start_time = time.time()
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[cp_callback], verbose=2)
time_taken = time.time() - start_time
print("Total time taken to fine-tune in seconds:", time_taken)

model.load_weights(checkpoint_path)

results = model.evaluate(test_ds, batch_size=batch_size)