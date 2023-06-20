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

"""
import matplotlib
#matplotlib.use("Agg")
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
"""
import io
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

from renderer import Renderer3D
from util import *

# this script replays the collected data (video + relative poses) in 3D and 2D
# it also calculates frame_paths (TODO: poses) and saves them

# EXAMPLE RUN: DATA_PATH="../collected_data/5/" ./replay.py

data_path = os.getenv("DATA_PATH")    # root dir
plog_poses = data_path + "poses.npy"  # global poses
fpath_log = data_path + "frame_paths.npy"
desire_dir = data_path + "desires.npy"

FRAME_PATHS = []  # NOTE: these are 2D for the time being

def figshow(fig):
  buf = io.BytesIO()
  pio.write_image(fig, buf)
  buf.seek(0)
  file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  cv2.imshow("Path Plot", img)


if __name__ == '__main__':
  renderer = Renderer3D(RW, RH)
  cap = cv2.VideoCapture(data_path+"video.mp4")
  poses = np.load(plog_poses)
  local_poses, local_path, local_orientations = get_relative_poses(poses)
  desires = one_hot_encode(np.load(desire_dir))

  fig = go.FigureWidget()
  fig.add_scatter()
  fig.update_layout(xaxis_range=[-50,50])
  fig.update_layout(yaxis_range=[0,50])

  h, w, c = None, None, None

  frame_id = 0
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - LOOKAHEAD
  prev_frame_time = 0
  new_frame_time = 0
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    if not (h and w and c):
      h, w, c = frame.shape

    print("[+] Frame %d"%(frame_id))
    # draw global path (3D) and calculate frame path (2D)
    renderer.draw(local_path[:frame_id+LOOKAHEAD], local_poses[frame_id])
    fpath = local_path[frame_id:frame_id+LOOKAHEAD]
    if len(fpath) == LOOKAHEAD:
      frame_path, frame_path_2d = get_frame_path(fpath)

      fig.data[0].x = frame_path_2d[:, 0]
      fig.data[0].y = frame_path_2d[:, 1]
      figshow(fig)

      FRAME_PATHS.append(frame_path_2d)
    else:
      break

    print("Frame Pose (Rt):")
    print(local_poses[frame_id])
    print("Frame Location (x,y,z):")
    print(local_path[frame_id])
    print("Frame Path Size:", frame_path.shape)
    draw_path(frame_path, frame)
    #cv2.imshow("2D DISPLAY", display_2D)

    # Image Display
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 1

    # display desire
    desire = desires[frame_id]
    desire_idx = np.argmax(desire)
    print("Desire:", desire_idx, "=>", DESIRE[desire_idx])

    org = (25, 55)
    color = (255, 0, 0)
    thickness = 2
    text = "DESIRE: %s" % (DESIRE[desire_idx])
    frame = cv2.putText(frame, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    # display FPS
    thickness = 2
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    print("FPS:", fps)
    frame = cv2.putText(frame,"FPS:"+fps, (w - 150, 25), font,
                      fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

    # display progress
    thickness = 2
    progress = ((frame_id+1) / total_frames) * 100
    text = f"{int(progress)}%"
    frame = cv2.putText(frame, text, (w - 150, h - 50), font,
                        fontScale, (0, 0, 255), thickness, cv2.LINE_AA)
    


    cv2.imshow("2D DISPLAY", frame)
    if cv2.waitKey(30) == ord('q'):
      break
    frame_id += 1

    print()
  
  cap.release()
  cv2.destroyAllWindows()

  FRAME_PATHS = np.stack(FRAME_PATHS)
  print(FRAME_PATHS)
  print(FRAME_PATHS.shape)
  np.save(fpath_log, FRAME_PATHS)
  print("[+] Frame Paths (2D) saved at", fpath_log)