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

data_path = "../collected_data/2/"   # CHANGE THIS
plog_path = data_path+"path.npy"  # CHANGE THIS


PATH = []
IMG_WIDTH = 1164
IMG_HEIGHT = 874
REC_TIME = 60 # recording length in seconds
LOOKAHEAD = 200

FRAME_TIME = 50

FOCAL = 910.0

# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array([
  [ 0.,  0.,  1.],
  [ 1.,  0.,  0.],
  [ 0.,  1.,  0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T

# aka 'K' aka camera_frame_from_view_frame
cam_intrinsics = np.array([
  [FOCAL,   0.,   IMG_WIDTH/2.],
  [  0.,  FOCAL,  IMG_HEIGHT/2.],
  [  0.,    0.,     1.]])

# aka 'K_inv' aka view_frame_from_camera_frame
eon_intrinsics_inv = np.linalg.inv(cam_intrinsics)


# TODO: display path on 3D as well (use pangolin)
def get_local_path(path):
  start_pos = path[0]
  local_path = []
  for i in range(len(path)):
    local_path.append(path[i] - start_pos)
  return np.array(local_path)

def img_from_device(device_path):
  in_shape = device_path.shape
  device_path = np.atleast_2d(device_path)
  path_view = np.einsum('jk, ik->ij', view_frame_from_device_frame, device_path)

  #path_view[path_view[:, 2] < 0] = np.nan  # TODO: commenting this out is a temp hack
  img_path = path_view/path_view[:,2:3]
  return img_path.reshape(in_shape)[:,:2]

def normalize(img_pts):
  # normalizes image coordinates
  # accepts single pt or array of pts
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0],1))))
  img_pts_normalized = eon_intrinsics_inv.dot(img_pts.T).T
  img_pts_normalized[(img_pts < 0).any(axis=1)] = np.nan
  return img_pts_normalized[:,:2].reshape(input_shape)

def denormalize(img_pts):
  # denormalizes image coordinates
  # accepts single pt or array of pts
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0],1))))
  img_pts_denormalized = cam_intrinsics.dot(img_pts.T).T
  img_pts_denormalized[img_pts_denormalized[:,0] > IMG_WIDTH] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,0] < 0] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,1] > IMG_HEIGHT] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,1] < 0] = np.nan
  return img_pts_denormalized[:,:2].reshape(input_shape)

def draw_path(path, img, width=1, height=1.2, fill_color=(128,0,255), line_color=(0,255,0)):
  img_points_norm = img_from_device(path) # TODO: this outputs NAN
  img_pts = denormalize(img_points_norm)
  valid = np.isfinite(img_pts).all(axis=1)
  img_pts = img_pts[valid].astype(int)

  # BUG: after a while, the beginning of the path disappears!!!
  print(len(img_pts))
  for i in range(1, len(img_pts)):
    #print(img_pts[i])
    cv2.circle(img, img_pts[i], 1, (0, 0, 255), -1)


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
    #print(path[frame_id])

    frame_path = local_path[frame_id:frame_id+LOOKAHEAD]
    print(frame_path.shape)
    draw_path(frame_path, frame)
    cv2.imshow("DISPLAY", frame)
    if cv2.waitKey(30) == ord('q'):
      break
    frame_id += 1
  
  cap.release()
  cv2.destroyAllWindows()
