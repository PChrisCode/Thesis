#@title Object Detector

import os
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import colorsys
from multiprocessing import Process
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.cluster import KMeans
from collections import defaultdict
import cv2
from sklearn.cluster import DBSCAN

CONSTANTS = {
  "TRAIN_DIRECTORY": "/content/gdrive/MyDrive/School/resources/raw_images/train",
  "TEST_DIRECTORY": "/content/gdrive/MyDrive/School/resources/raw_images/test",
  "INDEPENDENT_TEST_DIRECTORY": "/content/gdrive/MyDrive/School/resources/raw_images/independent_test",
  "IMAGE_WIDTH": 128,
  "IMAGE_HEIGHT": 128,
  "NUMBER_OF_ATTRIBUTES": 3,
  "COUNT_OF_LATENT_DIMENSIONS": 16,
  "BATCH_SIZE": 10,
  "CHECKPOINT_PATH_ENCODER": "/content/gdrive/MyDrive/School/resources/model/encoder_weights.h5",
  "CHECKPOINT_PATH_DECODER": "/content/gdrive/MyDrive/School/resources/model/decoder_weights.h5",
  "NUMBER_OF_EPOCHS": 6,
  "INPUT_SHAPE": (128, 128, 3),
  "IS_SEGMENTATION_ENABLED": True
}

class Autoencoder(Model):
  def __init__(self, latent_dimension):
    super(Autoencoder, self).__init__()
    self.latent_dimension = 12
    self.kernel_size = 3

    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(128, 128, 3), name="encoder_start"),
      layers.Conv2D(75, (3, 3), activation='relu', padding='same', strides=1, name="layer_10"),
      layers.Conv2D(50, (3, 3), activation='relu', padding='same', strides=1, name="layer_20"),
      layers.Conv2D(25, (3, 3), activation='sigmoid', padding='same', strides=1, name="layer_30"),
    ])
    self.encoder.summary()
    self.decoder = tf.keras.Sequential([
      layers.Input(shape=(128, 128, 25), name="decoder_start"),
      layers.Conv2DTranspose(25, kernel_size=3, strides=1, activation='relu', padding='same', name="layer_40"),
      layers.Conv2DTranspose(50, kernel_size=3, strides=1, activation='relu', padding='same', name="layer_50"),
      layers.Conv2DTranspose(75, kernel_size=3, strides=1, activation='relu', padding='same', name="layer_60"),
      layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same', name="layer_70")
    ])
    self.decoder.build(input_shape=(1, 128, 128, 3))
    self.decoder.summary()

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def decode_image_from_tfrecord(example_proto):
  feature_description = {
    'input': tf.io.FixedLenFeature([CONSTANTS["IMAGE_WIDTH"] * CONSTANTS["IMAGE_HEIGHT"] * CONSTANTS["NUMBER_OF_ATTRIBUTES"]], tf.float32),
    'output': tf.io.FixedLenFeature([CONSTANTS["IMAGE_WIDTH"] * CONSTANTS["IMAGE_HEIGHT"] * CONSTANTS["NUMBER_OF_ATTRIBUTES"]], tf.float32)
  }
  example = tf.io.parse_single_example(example_proto, feature_description)
  decoded_input = tf.reshape(example['input'], (CONSTANTS["IMAGE_WIDTH"], CONSTANTS["IMAGE_HEIGHT"], CONSTANTS["NUMBER_OF_ATTRIBUTES"]))
  decoded_output = tf.reshape(example['output'], (CONSTANTS["IMAGE_WIDTH"], CONSTANTS["IMAGE_HEIGHT"], CONSTANTS["NUMBER_OF_ATTRIBUTES"]))
  if (CONSTANTS["IS_SEGMENTATION_ENABLED"]):
    return [ decoded_input ], [ decoded_output ]
  else:
    return [ decoded_input ], [ decoded_input ]

def get_file_paths_by_directory(directory_path):
  file_paths = []
  for root, _, files in os.walk(directory_path):
    for file in files:
      file_paths.append(os.path.join(root, file))
  return file_paths

def is_file_exists(file_path):
  return os.path.exists(file_path)

def train_model():
  autoencoder = Autoencoder(CONSTANTS["COUNT_OF_LATENT_DIMENSIONS"])
  autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

  if is_file_exists(CONSTANTS["CHECKPOINT_PATH_ENCODER"]) and is_file_exists(CONSTANTS["CHECKPOINT_PATH_DECODER"]):
    autoencoder.encoder.build(input_shape=CONSTANTS["INPUT_SHAPE"])
    autoencoder.decoder.build(input_shape=(1, CONSTANTS["IMAGE_WIDTH"], CONSTANTS["IMAGE_HEIGHT"], 25))
    autoencoder.encoder.load_weights(CONSTANTS["CHECKPOINT_PATH_ENCODER"])
    autoencoder.decoder.load_weights(CONSTANTS["CHECKPOINT_PATH_DECODER"])
  else:
    train_file_paths = get_file_paths_by_directory(CONSTANTS["TRAIN_DIRECTORY"])

    train_dataset = tf.data.TFRecordDataset(train_file_paths, compression_type='GZIP')
    train_dataset = train_dataset.map(decode_image_from_tfrecord)

    test_file_paths = get_file_paths_by_directory(CONSTANTS["TEST_DIRECTORY"])
    test_dataset = tf.data.TFRecordDataset(train_file_paths, compression_type='GZIP')
    test_dataset = test_dataset.map(decode_image_from_tfrecord)

    autoencoder.fit(train_dataset, epochs=CONSTANTS["NUMBER_OF_EPOCHS"], batch_size=CONSTANTS["BATCH_SIZE"], shuffle=False, validation_data=test_dataset)
    autoencoder.encoder.save_weights(CONSTANTS["CHECKPOINT_PATH_ENCODER"])
    autoencoder.decoder.save_weights(CONSTANTS["CHECKPOINT_PATH_DECODER"])

  return autoencoder

def display_hls_image(hls_image):
    bgr_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)
    plt.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def show_image(hsv_image):
  rgb_image = np.zeros(CONSTANTS["INPUT_SHAPE"], dtype=np.uint8)
  for i in range(hsv_image.shape[0]):
      for j in range(hsv_image.shape[1]):
          r, g, b = hsv_image[i, j]
          rgb_image[i, j] = [int(r * 255.), int(g * 255.), int(b * 255.)]

  plt.imshow(rgb_image)
  plt.axis('off')
  plt.show()
  print("")

def main():
  model = train_model()

  test_file_paths = get_file_paths_by_directory(CONSTANTS["INDEPENDENT_TEST_DIRECTORY"])
  test_dataset = tf.data.TFRecordDataset(test_file_paths, compression_type='GZIP')
  test_dataset = test_dataset.map(decode_image_from_tfrecord)

  latent_vectors = model.encoder.predict(test_dataset)
  latent_vectors = latent_vectors * 255.

  images = model.decoder.predict(latent_vectors)
  images = images * 255.

  indicated_images = []
  for image in images:
    rows = []
    for row in image:
      pixels = []
      for pixel in row:
        r, g, b = pixel
        if (r > 239 and g < 11 and b < 11):
          pixels.append(1)
        else:
          pixels.append(0)
      rows.append(pixels)
    indicated_images.append(rows)

  first_image = indicated_images[0]
  eps = 2
  min_samples = 5
  dbscan = DBSCAN(eps=eps, min_samples=min_samples)
  labels = dbscan.fit_predict(first_image)
  unique_labels = np.unique(labels)
  print(f"Number of detected objects: {len(unique_labels)}")

  decoded_images = model.decoder.predict(latent_vectors)
  i = 0
  for image in decoded_images:
    show_image(image)
    i = i + 1

if __name__ == '__main__':
  main()