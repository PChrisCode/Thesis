#@title Data Generator

###########################
##### DATA GENERATOR #####
###########################

import os
import multiprocessing
import uuid
import random
import numpy as np
import colorsys
import tensorflow as tf
from PIL import Image

CONSTANTS = {
    "TARGET_OBJECTS_DROP_DIRECTORY": "/content/gdrive/MyDrive/School/resources/target_objects",
    "TRAIN_DATA_DROP_DIRECTORY": "/content/gdrive/MyDrive/School/resources/raw_images/train",
    "TEST_DATA_DROP_DIRECTORY": "/content/gdrive/MyDrive/School/resources/raw_images/test",
    "INDEPENDENT_DATA_DROP_DIRECTORY": "/content/gdrive/MyDrive/School/resources/raw_images/independent_test",
    "TRAIN_DATA_BACKUP_DROP_DIRECTORY": "/content/gdrive/MyDrive/School/resources/raw_images/train_backup",
    "TEST_DATA_BACKUP_DROP_DIRECTORY": "/content/gdrive/MyDrive/School/resources/raw_images/test_backup",
    "INDEPENDENT_DATA_BACKUP_DROP_DIRECTORY": "/content/gdrive/MyDrive/School/resources/raw_images/independent_backup",
    "IMAGE_WIDTH": 128,
    "IMAGE_HEIGHT": 128,
    "OBJECT_SIZE_MIN": 12,
    "OBJECT_SIZE_MAX": 20,
    "MIN_NUMBER_OF_OBJECTS": 8,
    "MAX_NUMBER_OF_OBJECTS": 12,
    "IS_DEBUG_LOG_ENABLED": False
}

def generate_data_parallel(count_of_entities, drop_directory, backup_drop_directory):
  directory = get_absolute_path_by_relative_path(drop_directory)
  create_directory_if_not_exists(directory)
  create_directory_if_not_exists(backup_drop_directory)
  count_of_generated_files = get_count_of_files_in_directory(directory)
  count_of_entities = count_of_entities - count_of_generated_files
  print("GENERATING " + str(count_of_entities) + " DATA")
  run_parallel(generate_data, count_of_entities, (drop_directory,backup_drop_directory))

def get_absolute_path_by_relative_path(relative_path):
    script_directory = os.path.abspath('.')
    absolute_path = os.path.abspath(os.path.join(script_directory, relative_path))
    return absolute_path

def create_directory_if_not_exists(folder_name):
  if not os.path.exists(folder_name):
    os.mkdir(folder_name)

def get_count_of_files_in_directory(directory_path):
  return len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])

def run_parallel(method, iterations, parameter):
  if __name__ == '__main__':
    max_number_of_cpus = multiprocessing.cpu_count()
    min_number_of_cpus = max(1, int(max_number_of_cpus * 0.9))
    number_of_processes = max(min_number_of_cpus, max_number_of_cpus)
    if number_of_processes >= 1:
      pool = multiprocessing.Pool(processes=number_of_processes)
      arguments = [parameter] * iterations
      pool.map(method, arguments)
      pool.close()
      pool.join()

def generate_data(arguments):
  drop_directory = get_absolute_path_by_relative_path(arguments[0])
  backup_drop_directory = get_absolute_path_by_relative_path(arguments[1])
  count_of_target_objects = random.randint(CONSTANTS["MIN_NUMBER_OF_OBJECTS"], CONSTANTS["MAX_NUMBER_OF_OBJECTS"])
  template_objects = create_template_objects(count_of_target_objects)

  input_data = Image.new("RGBA", (CONSTANTS["IMAGE_WIDTH"], CONSTANTS["IMAGE_HEIGHT"]), "white")
  output_data = Image.new("RGBA", (CONSTANTS["IMAGE_WIDTH"], CONSTANTS["IMAGE_HEIGHT"]), "white")

  input_data = convert_pil_image_to_np_array(input_data)
  output_data = convert_pil_image_to_np_array(output_data)

  for object in template_objects:
    position = add_image_to_wallpaper(input_data, object, is_collision_allowed=False, is_write_object_existence=False, target_x=None, target_y=None)
    _ = add_image_to_wallpaper(output_data, object, is_collision_allowed=False, is_write_object_existence=True, target_x=position[0], target_y=position[1])

  base_file_name = str(uuid.uuid4())

  input_data = remove_subdimension(input_data, 5)
  input_data = remove_subdimension(input_data, 3)
  if (CONSTANTS["IS_DEBUG_LOG_ENABLED"]):
    save_png_image(input_data, drop_directory, base_file_name + "-input", False)
  save_png_image(input_data, backup_drop_directory, base_file_name + "-input", False)
  normalized_input_data = normalize_data(input_data)

  output_data = remove_subdimension(output_data, 5)
  output_data = remove_subdimension(output_data, 3)
  if (CONSTANTS["IS_DEBUG_LOG_ENABLED"]):
    save_png_image(output_data, drop_directory, base_file_name + "-output", False)
  save_png_image(output_data, backup_drop_directory, base_file_name + "-output", False)
  normalized_output_data = normalize_data(output_data)

  object_title = str(uuid.uuid4())
  create_directory_if_not_exists(drop_directory)
  object_path = drop_directory + "/" + object_title + ".tfrecord.gz"
  save_data_to_gzip_tfrecord(normalized_input_data, normalized_output_data, object_path)

def create_template_objects(count_of_entities):
  template_objects = []
  possible_rotations = [0, 90, 180, 270]
  for _ in range(count_of_entities):
    template_object = get_random_template_object()
    rotation_index = random.randint(0, len(possible_rotations) - 1)
    selected_rotation = possible_rotations[rotation_index]
    rotated_object = template_object.rotate(selected_rotation)
    template_objects.append(rotated_object)
  return template_objects

def get_random_template_object():
  objects = get_files_by_directory(get_absolute_path_by_relative_path(CONSTANTS["TARGET_OBJECTS_DROP_DIRECTORY"]))
  image_index = random.randint(0, len(objects) - 1)
  image_path = objects[image_index]
  original_template_image = Image.open(image_path)
  size = random.randint(CONSTANTS["OBJECT_SIZE_MIN"], CONSTANTS["OBJECT_SIZE_MAX"])
  template_image_width = size
  template_image_height = size
  template_image = original_template_image.resize((template_image_width, template_image_height), Image.LANCZOS)
  return template_image

def get_files_by_directory(directory_path):
  return [os.path.abspath(os.path.join(directory_path, f)) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

def convert_pil_image_to_np_array(image):
  image_rgba = image.convert('RGBA')
  image_array = np.array(image_rgba)
  image_array = add_subdimension(image_array, 0.0)
  image_array = add_subdimension(image_array, False)
  return image_array

def add_subdimension(arr, default_value):
  number_of_new_subdimensions = arr.shape[2] + 1
  new_shape = (arr.shape[0], arr.shape[1], number_of_new_subdimensions)
  new_arr = np.zeros(new_shape)
  new_arr[:, :, :(number_of_new_subdimensions - 1)] = arr
  new_arr[:, :, (number_of_new_subdimensions - 1)] = default_value
  return new_arr

def add_image_to_wallpaper(wallpaper, image, is_collision_allowed=False, is_write_object_existence=False, target_x=None, target_y=None):
  image_array = np.array(image)
  image_height, image_width, _ = image_array.shape

  target_x_coord = None
  target_y_coord = None

  if (target_x == None and target_y == None):
    max_x = wallpaper.shape[1] - image_width
    max_y = wallpaper.shape[0] - image_height
    if max_x < 0 or max_y < 0:
      raise ValueError("Image dimensions are larger than wallpaper dimensions.")

    target_x = random.randint(0, max_x)
    target_y = random.randint(0, max_y)

    if not is_collision_allowed:
      while is_collision_detected(wallpaper, image_array, target_x, target_y):
        target_x = random.randint(0, max_x)
        target_y = random.randint(0, max_y)
    target_x_coord, target_y_coord = target_x, target_y
  else:
    target_x_coord, target_y_coord = target_x, target_y

  object_existence = 0.

  for y in range(image_height):
    for x in range(image_width):
      img_x = x
      img_y = y

      next_x = target_x_coord + x
      next_y = target_y_coord + y

      rgba = image_array[img_y, img_x]
      red = rgba[0]
      green = rgba[1]
      blue = rgba[2]

      alpha = rgba[3]
      current_pixel = wallpaper[next_y, next_x]
      if current_pixel[4] > 0. or (red == 0 and green == 0 and blue == 0 and alpha == 0): # Skip the parts where object indicator cannot be found or no object data can be found
        continue
      if (alpha == 0 and (current_pixel[0] == 255 and current_pixel[1] == 255 and current_pixel[2] == 255)): # if the alpha is 0 and the pixel is white then we have to add a wallpaper pixel
        wallpaper[next_y, next_x] = (255, 255, 255, 255, 0., True)
        continue

      if (is_write_object_existence):
        wallpaper[next_y, next_x] = (255, 0, 0, 255, object_existence, True)
      else:
        wallpaper[next_y, next_x] = (red, green, blue, 255, object_existence, True)

  return (target_x_coord, target_y_coord)

def is_collision_detected(wallpaper, image_array, target_x, target_y):
  image_height, image_width, _ = image_array.shape
  for y in range(image_height):
    for x in range(image_width):
      next_x = target_x + x
      next_y = target_y + y
      wallpaper_pixel = wallpaper[next_y, next_x]
      if (wallpaper_pixel[5] == True):
        return True
  return False

def remove_subdimension(arr, subdimension_index):
  if subdimension_index < 0 or subdimension_index >= arr.shape[2]:
      raise ValueError("Invalid subdimension_index")
  result = np.delete(arr, subdimension_index, axis=2)
  return result

def normalize_data(data):
  white = 255. / 255.
  result = np.full(shape=(CONSTANTS["IMAGE_WIDTH"], CONSTANTS["IMAGE_HEIGHT"], 3), fill_value=white)
  for x in range(len(data)):
    for y in range(len(data[x])):
      red = data[y, x][0] / 255.
      green = data[y, x][1] / 255.
      blue = data[y, x][2] / 255.
      result[y, x] = (red, green, blue)
  return result

def save_png_image(data, drop_directory, file_name, is_object_mark_enabled):
  image_path = get_absolute_path_by_relative_path(drop_directory) + "/" + file_name + ".png"
  save_png_from_array(data, image_path, is_object_mark_enabled)

def save_data_to_gzip_tfrecord(input_data, output_data, tfrecord_file_path):
  with tf.io.TFRecordWriter(tfrecord_file_path, options='GZIP') as writer:
    feature_description = {
      'input': tf.train.Feature(float_list=tf.train.FloatList(value=input_data.flatten())),
      'output': tf.train.Feature(float_list=tf.train.FloatList(value=output_data.flatten())),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_description))
    writer.write(example.SerializeToString())

def save_png_from_array(array, output_filename, is_object_mark_enabled):
  image = Image.new("RGB", (CONSTANTS["IMAGE_WIDTH"], CONSTANTS["IMAGE_HEIGHT"]))
  for x in range(array.shape[0]):
    for y in range(array.shape[1]):
      rgbo = array[y, x]
      wallpaper_rgb = image.getpixel((x, y))
      if (is_object_mark_enabled):
        mark = rgbo[3]
        if (wallpaper_rgb[0] == 255 and wallpaper_rgb[1] == 255 and wallpaper_rgb[2] == 255):
          continue
        if mark == 0.5:
          pixel_color = (255, 255, 0)
          image.putpixel((x, y), pixel_color)
        elif mark == 1.0:
          pixel_color = (255, 0, 0)
          image.putpixel((x, y), pixel_color)
        else:
          pixel_color = (255, 255, 255)
          image.putpixel((x, y), pixel_color)
      else:
        pixel_color = (int(rgbo[0]), int(rgbo[1]), int(rgbo[2]))
        image.putpixel((x, y), pixel_color)
  image.save(output_filename, "PNG")

def main():
  train_drop_directory = get_absolute_path_by_relative_path(CONSTANTS["TRAIN_DATA_DROP_DIRECTORY"])
  generate_data_parallel(30, train_drop_directory, CONSTANTS["TRAIN_DATA_BACKUP_DROP_DIRECTORY"])

  test_drop_directory = get_absolute_path_by_relative_path(CONSTANTS["TEST_DATA_DROP_DIRECTORY"])
  generate_data_parallel(10, test_drop_directory, CONSTANTS["TEST_DATA_BACKUP_DROP_DIRECTORY"])

  independent_drop_directory = get_absolute_path_by_relative_path(CONSTANTS["INDEPENDENT_DATA_DROP_DIRECTORY"])
  generate_data_parallel(5, independent_drop_directory, CONSTANTS["INDEPENDENT_DATA_BACKUP_DROP_DIRECTORY"])

if __name__ == '__main__':
  main()