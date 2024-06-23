#@title Template matching

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

image = cv2.imread('/content/gdrive/MyDrive/School/resources/raw_images/independent_backup/98caea16-f75b-459c-b7e1-3d00de57de41-input.png', cv2.IMREAD_COLOR )
template = cv2.imread('/content/gdrive/MyDrive/School/resources/target_objects/Sunflower-09.png', cv2.IMREAD_COLOR)
template = cv2.resize(template, (16, 16))

h, w = template.shape[:2]
method = cv2.TM_CCOEFF_NORMED
threshold = 0.7
max_val = 1

while max_val > threshold:
  res = cv2.matchTemplate(image, template, method)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

  image[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w, :] = [0, 0, 255]

cv2_imshow(image)