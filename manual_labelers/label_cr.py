#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np

# (3, 960, 1280) inputs images
W = 1280
H = 960

# TODO: implement timestamps (so we can go back and forth wherever we want in the video, this might need extra work with the line indices so they can match the frames)
# 1 = crossroad, 0 = no_crossroad
CROSSROAD = {0: "no-crossroad", 1: "crossroad"}

if __name__ == '__main__':
  data_path = os.getenv("DATA_PATH")
  video_path = data_path + "video.mp4"
  cr_path = data_path + "crossroads.npy"

  #cap = cv2.VideoCapture(data_path+filename)
  cap = cv2.VideoCapture(video_path)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  labels = []
  
  idx = 0
  while True:
    ret, frame = cap.read()
    
    if ret:
      frame = cv2.resize(frame, (W,H))
      print("Frame", idx, "/", total_frames)
      print(frame.shape)
      cv2.imshow('frame', frame)
      idx += 1

      key = cv2.waitKey(0)
      # if key pressed is 'c' then crossroad detected, if key is 'q' stop, if key is other continue (no crossroad)
      label = 0
      if key & 0xff == ord('c'):
        label = 1
      elif key & 0xff == ord('q'):
        break
      labels.append([label])
      print("Label:", label, "=>", CROSSROAD[label])

    else:
      break

  labels = np.array(labels)
  print(labels)
  cap.release()
  cv2.destroyAllWindows()

  # save to file
  np.save(cr_path, labels)
  print("Crossroad labels saved at:", cr_path)

