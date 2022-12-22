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

# TODO: this script should replay carla collected videos and draw paths (also dump the 2D paths from the image)

PATH = []
IMG_WIDTH = 1164
IMG_HEIGHT = 874
REC_TIME = 60 # recording length in seconds
LOOKAHEAD = 200

FRAME_TIME = 50

data_path = "../collected_data/2/"   # CHANGE THIS
plog_path = data_path+"path.npy"  # CHANGE THIS

def get_local_path(path):
  start_pos = path[0]
  local_path = []
  for i in range(len(path)):
    local_path.append(path[i] - path[0])
  return np.array(local_path)

def draw_path(path, img, width=1, height=1.2, fill_color=(128,0,255), line_color=(0,255,0)):
  pass

# TODO: process path to be relative to car and project it from 3D to 2D display
if __name__ == '__main__':
  cap = cv2.VideoCapture(data_path+"video.mp4")
  path = np.load(plog_path)
  local_path = get_local_path(path)

  frame_id = 0
  while True:
    ret, frame = cap.read()   
    if not ret:
      break

    print("[+] Frame %d"%(frame_id))
    print(path[frame_id])

    draw_path(local_path[frame_id:frame_id+LOOKAHEAD], frame)
    cv2.imshow("DISPLAY", frame)
    if cv2.waitKey(30) == ord('q'):
      break
    frame_id += 1
  
  cap.release()
  cv2.destroyAllWindows()
