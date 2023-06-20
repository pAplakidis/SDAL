#!/usr/bin/env python3
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import utils

from train_utils import *

base_dir = "data/"  # base directory for training dataset

"""
CLASSES:
[[ 32  32  64]  : drivable area
 [  0   0 255]  : lane-lines
 [  0 255   0]  : other cars
 [255   0 204]  : self-car
 [ 96 128 128]] : non-driveable area
"""

# TODO: other cars (GREEN) class is missing in both processed groundtruth and prediction
# + the model skips laneline (RED) class
def segnet_to_rgb(img, classes):
  # NOTE: 5,360,480 means that each channel of the 5 is a probability the pixel belongs in a specific class
  indices = torch.argmax(img, dim=0)

  out_img = []
  for i in range(indices.shape[0]):
    tmp = []
    for j in range(indices.shape[1]):
      tmp.append(classes[indices[i][j]])
    out_img.append(tmp)

  out_img = np.array(out_img, dtype=np.uint8)
  #print(out_img)
  #print(out_img.shape)
  return out_img


def overlay_mask(img, mask):
  """
  def to_png(img, a):
    fin_img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    b, g, r, alpha = cv2.split(fin_img)
    alpha = a
    fin_img[:,:, 0] = img[:,:,0]
    fin_img[:,:, 1] = img[:,:,1]
    fin_img[:,:, 2] = img[:,:,2]
    fin_img[:,:, 3] = alpha[:,:]
    return fin_img
  """

  #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # TODO: use the grayscale of this image
  for alpha in np.arange(0, 1.1, 0.1)[::-1]:
    overlay = img.copy()
    out = img.copy()
    overlay = cv2.bitwise_or(img, mask)

    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out

def view_net_result(origin_img, pred_img, classes, gt_img=None):
  cv2.imshow('image', origin_img)
  gt = segnet_to_rgb(torch.tensor(gt_img), classes)
  cv2.imshow('ground truth', gt)
  segmented_img = segnet_to_rgb(pred_img, classes)
  cv2.imshow('prediction', segmented_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def get_only_object(img, mask, back_img):
  fg = cv2.bitwise_or(img, img, mask=mask)        
  # invert mask
  mask_inv = cv2.bitwise_not(mask)    
  fg_back_inv = cv2.bitwise_or(back_img, back_img, mask=mask_inv)
  final = cv2.bitwise_or(fg, fg_back_inv)
  return final

# TODO: view network's output overlapping original image
def view_overlap_net_result(origin_img, pred_img, gt_img=None):
  pass


if __name__ == "__main__":
  classes = np.load("data/classes.npy")
  print(len(classes))
  dataset = CommaDataset(base_dir, classes, for_net=False)
  loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
  print(len(dataset))

  for i in range(1):
    samp = dataset[random.randint(0, len(dataset))]
    img, mask = samp['image'], samp['mask']
    overlay = overlay_mask(img, mask)
    print(img.shape)
    print(mask.shape)
    cv2.imshow('image', img)
    cv2.imshow('mask', mask)
    cv2.imshow('overlay', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  """
  for i_batch, sample_batched in enumerate(loader):
    print(i_batch, sample_batched['image'].size(), sample_batched['mask'].size())

    if i_batch == 3:
      plt.figure()
      show_imgmask_batch(sample_batched)
      plt.axis('off')
      plt.ioff()
      plt.show()
      break
  """
