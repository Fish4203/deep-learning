import tensorflow as tf
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

nclasses = 10
nsamples = 3000000
bsize = nsamples//20
inp_units = 100
mod = tf.keras.Sequential([tf.keras.layers.InputLayer(inp_units), tf.keras.layers.Dense(2500, activation='relu'), tf.keras.layers.Dense(2500, activation='relu'), tf.keras.layers.Dense(250, activation='relu'), tf.keras.layers.Dense(nclasses, activation='softmax')])
mod.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

inpt = np.random.rand(nsamples,inp_units)
gtt = np.random.randint(0,nclasses-1,nsamples)
dset = tf.data.Dataset.from_tensor_slices((inpt,gtt)).batch(bsize)

mod.fit(dset, epochs = 20)