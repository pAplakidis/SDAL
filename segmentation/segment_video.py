#!/usr/bin/env python3
import os
import cv2

from util import *
from train_utils import *
from model import *

# EXAMPLE RUN: DATA_DIR="../collected_data/5/" ./segment_video.py

if __name__ == "__main__":
  #data_dir = "../collected_data/2/"
  data_dir = os.getenv("DATA_DIR")
  if data_dir == None:
    print("Specify a path for data")
    exit(0)
  video_path = data_dir+"video.mp4"
  output_dir = data_dir + "segnet_out/"
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  img_dir = output_dir + "imgs/"
  mask_dir = output_dir + "masks/"
  if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
    os.makedirs(img_dir)
    os.makedirs(mask_dir)

  #model_path = "models/segnet_adam1e-3.pth"
  model_path = "models/segnet_SGD_weighted_CEL.pth"

  def segment_video():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    classes = np.load("data/classes.npy")
    print("[*] Classes Found:", classes.shape)
    print(classes)

    cap = cv2.VideoCapture(video_path)

    with torch.no_grad():
      model = SegNet(3, len(classes))
      model = load_model(model_path, model)
      model.to(device)
      model.eval()

      idx = 0
      total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      while True:
        ret, frame = cap.read()
        if ret:
          print("[+] Processing frame %d/%d"%(idx+1, total_frames))
          img = cv2.resize(frame, (W,H))
          img_in = np.moveaxis(img, -1, 0)
          X = torch.tensor([img_in, img_in]).float().to(device)

          out = model(X)
          mask = segnet_to_rgb(out[0], classes)

          #cv2.imshow('frame', frame)
          #cv2.imshow('mask', mask)
          #if cv2.waitKey(0) & 0xff == ord('q'):
          #  break

          img_path = img_dir + str(idx) + ".png"
          mask_path = mask_dir + str(idx) + ".png"

          cv2.imwrite(img_path, img)
          cv2.imwrite(mask_path, mask)

          idx += 1
        else:
          break

    cap.release()
    cv2.destroyAllWindows()

  # TODO: write scripts that extract labels from mask


  if __name__ == "__main__":
    segment_video()
