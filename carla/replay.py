#!/usr/bin/env python3
import random
import math
import glob
import sys
import os
import time
import cv2
import numpy as np
from multiprocessing import Process, Queue
from threading import Thread

from renderer import Renderer3D
from util import *

# TODO: this script should replay carla collected videos and draw paths (also dump the 2D paths from the image)

data_path = "../collected_data/2/"   # CHANGE THIS
plog_path = data_path+"path.npy"  # CHANGE THIS

def draw_path(path, img, width=1, height=1.2, fill_color=(128,0,255), line_color=(0,255,0)):
  # TODO: debug 3D to 2D convertion
  img_points_norm = img_from_device(path) # TODO: this outputs NAN
  img_pts = denormalize(img_points_norm)
  valid = np.isfinite(img_pts).all(axis=1)
  img_pts = img_pts[valid].astype(int)

  print(len(img_pts))
  for i in range(1, len(img_pts)):
    #print(img_pts[i])
    cv2.circle(img, img_pts[i], 1, (0, 0, 255), -1)


if __name__ == '__main__':
  renderer = Renderer3D(RW, RH)
  cap = cv2.VideoCapture(data_path+"video.mp4")
  path = np.load(plog_path)
  local_path = get_local_path(path)

  frame_id = 0
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    print("[+] Frame %d"%(frame_id))
    #print(path[frame_id])

    renderer.draw(local_path)
    frame_path = local_path[frame_id:frame_id+LOOKAHEAD]
    print(frame_path.shape)
    draw_path(frame_path, frame)
    cv2.imshow("2D DISPLAY", frame)
    if cv2.waitKey(30) == ord('q'):
      break
    frame_id += 1
  
  cap.release()
  cv2.destroyAllWindows()
