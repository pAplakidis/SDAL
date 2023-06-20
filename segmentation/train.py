#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split

from model import *
from util import *
from train_utils import *

# EXAMPLE RUN: TRAIN=1 BS=2 EPOCHS=100 LR=1e-3 WRITER_PATH="runs/overfit" MODEL_PATH="models/segnet.pth" ./train.py

# TODO: train with SGD(lr=0.1, momentum=0.9), just like the paper
# TODO: add weight to the loss of different classes based on frequency (high frequency => weight<1)

TRAIN = os.getenv("TRAIN")
if TRAIN == None or TRAIN == '1':
  TRAIN = True
elif TRAIN == '0':
  TRAIN = False
print("[+] Training Mode:", TRAIN)

BS = os.getenv("BS")
if BS == None:
  BS = 2  # NOTE: this is the max batch size my home-PC can handle, paper used 12
else:
  BS = int(BS)
print("[+] Using Batch Size:", BS)

EPOCHS = os.getenv("EPOCHS")
if EPOCHS != None:
  EPOCHS = int(EPOCHS)
else:
  EPOCHS = 100
print("[+] Max epochs:", EPOCHS)

LR = os.getenv("EPOCHS")
if LR != None:
  LR = float(LR)
else:
  LR = 1e-3
print("[+] Learning Rate:", LR)

model_path = os.getenv("MODEL_PATH")
if model_path == None:
  model_path = "models/segnet.pth"
print("[+] Model save path:", model_path)

writer_path = os.getenv("WRITER_PATH")


if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  classes = np.load("data/classes.npy")
  print("[+] Classes Found:", classes.shape)
  print(classes)

  # get data
  dataset = CommaDataset(base_dir, classes)
  train_set, val_set = random_split(dataset, [int(len(dataset)*0.7)+1, int(len(dataset)*0.3)])
  train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_set, batch_size=BS, shuffle=True, num_workers=0)

  # define model and train
  in_samp = dataset[0]['image']
  out_samp = dataset[0]['mask']
  in_ch, out_ch = in_samp.shape[0], out_samp.shape[0]
  model = SegNet(in_ch, out_ch)
  if TRAIN:
    if writer_path:
      trainer = Trainer(device, model, train_loader, val_loader, model_path, writer_path=writer_path)
    else:
      trainer = Trainer(device, model, train_loader, val_loader, model_path)
    model = trainer.train(epochs=EPOCHS, lr=LR)  # NOTE: lr=1e-3 seems to be optimal
  else:
    model = load_model(model_path, model)
    model.to(device)

  # view some images to examine the model's progress
  with torch.no_grad():
    model.eval()
    print("[*] Training images preview")
    for i in range(5):
      samp = train_set[random.randint(0, len(train_set))]
      img, mask = samp['image'], samp['mask']
      out_img = np.moveaxis(img, 0, -1)
      X = torch.tensor([img, img]).float().to(device)
      pred = model(X)
      view_net_result(out_img, pred[0], classes, gt_img=mask)

    print("[*] Evaluation images preview")
    for i in range(5):
      samp = val_set[random.randint(0, len(val_set))]
      img, mask = samp['image'], samp['mask']
      out_img = np.moveaxis(img, 0, -1)
      X = torch.tensor([img, img]).float().to(device)
      pred = model(X)
      view_net_result(out_img, pred[0], classes, gt_img=mask)
