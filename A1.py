import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

print(tf.__version__)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)


imageFileNames = tf.data.Dataset.list_files('./images/*', shuffle=True)

labelSet = set()
labelDict = {}
with open("./dev_data_2025.csv", "r") as f:
    for imgLabel in csv.DictReader(f):
        labelSet.add(imgLabel['label'])
        labelDict[imgLabel['imageID']] = [
            imgLabel['label'], 
            imgLabel['cell_shape'], 
            imgLabel['nucleus_shape'], 
            imgLabel['cytoplasm_vacuole']
        ]



# Making the datasets


def getLabel(path):
    path = path.numpy().decode("utf-8")
    key = os.path.basename(path)[:9]
    return tf.argmax(labelDict[key])

def getImage(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize_with_crop_or_pad(img, 360, 360)

def process_path(file_path):    
    label = tf.py_function(func=getLabel, inp=[file_path], Tout=tf.uint32)
    label.set_shape([None])

    img = tf.py_function(func=getImage, inp=[file_path], Tout=tf.uint8)
    img.set_shape([None, None, 3])

    return img, label


valSize = int(imageFileNames.cardinality().numpy() * 0.2)

trainData = imageFileNames \
  .skip(valSize) \
  .map(process_path, num_parallel_calls=tf.data.AUTOTUNE) \
  .cache() \
  .batch(100) \
  .prefetch(buffer_size=tf.data.AUTOTUNE)
valData = imageFileNames \
  .take(valSize) \
  .map(process_path, num_parallel_calls=tf.data.AUTOTUNE) \
  .cache() \
  .batch(100) \
  .prefetch(buffer_size=tf.data.AUTOTUNE)

print(tf.data.experimental.cardinality(trainData).numpy())
print(tf.data.experimental.cardinality(valData).numpy())

# Visuals
plt.figure(figsize=(10, 10))
i = 0
for image, label in trainData.take(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image.numpy().astype("uint8"))
  plt.title(str(label.numpy()))
  plt.axis("off")
  i += 1



num_classes = len(labelSet)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(360, 360, 3)),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(num_classes)  
])

model.compile(
    optimizer='Adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()


model.fit(
  trainData,
  validation_data=valData,
  epochs=3
)

