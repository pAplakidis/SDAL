#!/usr/bin/env python3
import sys
import cv2
import numpy as np

# (3, 960, 1280) inputs images
W = 1280
H = 960

# TODO: implement timestamps (so we can go back and forth wherever we want in the video, this might need extra work with the line indices so they can match the frames)
DESIRE = {0: "forward",
          1: "right",
          2: "left"}

# for every frame/line output the label in a txt file
if __name__ == '__main__':
  # NOTE: only .mp4 videos
  base_path = sys.argv[1]
  video_path = base_path + "video.mp4"
  label_path = base_path + "desires.npy"

  cap = cv2.VideoCapture(video_path)
  labels = []
  
  idx = 0
  while True:
    ret, frame = cap.read()
    
    if ret:
      frame = cv2.resize(frame, (W,H))
      print("Frame", idx)
      print(frame.shape)
      cv2.imshow('frame', frame)
      idx += 1

      key = cv2.waitKey(0)
      # if key pressed is 'c' then crossroad detected, if key is 'q' stop, of key is other continue (no crossroad)
      if key & 0xff == ord('w'):
        label = 0
        labels.append(label)
      elif key & 0xff == ord('d'):
        label = 1
        labels.append(label)
      elif key & 0xff == ord('a'):
        label = 2
        labels.append(label)
      elif key & 0xff == ord('q'):
        break
      else:
        label = 0
        labels.append(label)
      print(DESIRE[label])
    else:
      break

  labels = np.array(labels)
  print(labels)
  cap.release()
  cv2.destroyAllWindows()

  # save to file
  out_path = label_path
  np.save(out_path, labels)
  print("Labels written to", out_path)

