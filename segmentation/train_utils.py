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

import model

# net input resolution
W = 480
H = 360

# display resolution
d_W = 1920 // 2
d_H = 1080 // 2

tb_path = "runs/single_batch_overfit"

# intersection-over-union
def iou_coef(y_true, y_pred, smooth=1):
  intersection = torch.sum(torch.abs(y_true * y_pred), axis=[1,2,3])
  union = torch.sum(y_true, [1,2,3]) + torch.sum(y_pred, [1,2,3]) - intersection
  iou = torch.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou.item()


# custom loader for comma10k dataset
class CommaDataset(Dataset):
  def __init__(self, base_dir, classes, for_net=True):
    super(Dataset, self).__init__()
    self.for_net = for_net
    self.classes = classes
    self.base_dir = base_dir
    self.img_dir = base_dir + "imgs/"
    self.mask_dir = base_dir + "masks/"
    self.img_list = os.listdir(self.img_dir)
    self.mask_list = os.listdir(self.img_dir)

  def __len__(self):
    img_len = len(self.img_list)
    mask_len = len(self.mask_list)
    assert img_len == mask_len
    return img_len

  def __getitem__(self, idx):
    img = cv2.imread(self.img_path(self.img_list[idx]))
    img = cv2.resize(img, (W,H))
    if self.for_net:
      img = np.moveaxis(img, -1, 0)
    mask = cv2.imread(self.mask_path(self.mask_list[idx]))
    mask_original = cv2.resize(mask, (W,H))

    """
    print(mask.shape)
    classes = []
    for i in range(mask.shape[0]):
      for j in range(mask.shape[1]):
        if list(mask[i][j]) not in classes:
          print(mask[i][j])
          classes.append(list(mask[i][j]))
    classes = np.array(classes)
    print(classes.shape)
    np.save("data/classes.npy", classes)
    exit(0)
    """

    if self.for_net:
      mask = self.onehot(mask_original)
      mask = np.moveaxis(mask, -1, 0)
    return {'image': img, 'mask': mask, "mask_original": mask_original}

  def img_path(self, f):
    return self.img_dir+f

  def mask_path(self, f):
    return self.mask_dir+f

  def onehot(self, img):
    out_shape = (img.shape[0], img.shape[1], self.classes.shape[0])
    out = np.zeros(out_shape)

    for c in range(self.classes.shape[0]):
      label = np.nanmin(self.classes[c] == img, axis=2)
      out[:, :, c] = label

    return out


def show_imgmask_batch(sample_batched):
  img_batch, mask_batch = sample_batched['image'], sample_batched['mask']
  batch_size = len(img_batch)
  img_size = img_batch.size(2)
  grid_border_size = 2

  grid = utils.make_grid(img_batch)
  #grid = utils.make_grid(mask_batch)
  plt.imshow(grid.numpy().transpose((1,2,0)))


class Trainer:
  def __init__(self, device, model, train_loader, val_loader, model_path, eval=True, early_stop=False, writer_path=None):
    self.eval = eval
    self.early_stop = early_stop
    self.model_path = model_path
    if not writer_path:
      writer_path = tb_path
    print("[+] Tensorboard output path:", writer_path)

    self.writer = SummaryWriter(writer_path)
    self.device = device
    print("[+] Device:", self.device)
    self.model = model.to(self.device)
    self.train_loader = train_loader
    self.val_loader = val_loader

  def save_checkpoint(state, path):
    torch.save(state, path)
    print("Checkpoint saved at", path)

  def train(self, epochs=100, lr=1e-1, path=None):
    self.model.train()
    class_weights = torch.FloatTensor([0.3, 0.3, 0.2, 0.1, 0.1]).to(self.device)  # TODO: finetune class weights
    loss_func = nn.CrossEntropyLoss(weight=class_weights)
    optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
    #optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    tb_images = next(iter(self.train_loader))['image'].float().to(self.device)
    tb_masks = next(iter(self.train_loader))['mask'].float().to(self.device)
    self.writer.add_graph(self.model, tb_images)
    #self.writer.add_graph(self.model, tb_masks)

    def eval(val_losses, val_iou, train=False):
      print("[+] Evaluating ...")
      with torch.no_grad():
        try:
          self.model.eval()
          l_idx = 0
          for i_batch, sample_batched in enumerate((t:= tqdm(self.val_loader))):
            X = sample_batched['image'].float().to(self.device)
            Y = sample_batched['mask'].float().to(self.device)
            Y_idx = torch.argmax(Y, dim=1)

            out = self.model(X)
            loss = loss_func(out, Y_idx)
            iou_acc = iou_coef(Y, out)

            if not train:
              self.writer.add_scalar('evaluation loss', loss.item(), l_idx)
              self.writer.add_scalar('evaluation IOU', iou_acc, l_idx)
            val_losses.append(loss.item())
            val_iou.append(iou_acc)
            t.set_description("%d/%d: Batch Loss: %.2f, IOU: %.2f"%(i_batch+1, len(self.val_loader), loss.item(), iou_acc))
            l_idx += 1
            #break

        except KeyboardInterrupt:
          print("[~] Evaluation stopped by user")
      print("[+] Evaluation Done")
      return val_losses, val_iou

    losses = []
    px_accuracies = []  # TODO: per class accuracies
    iou_accuracies = []
    vlosses = []

    try:
      print("[+] Training ...")
      l_idx = 0
      for epoch in range(epochs):
        self.model.train()
        print("[+] Epoch %d/%d"%(epoch+1,epochs))
        epoch_losses = []
        epoch_pxaccuracies = []
        epoch_iouacc = []
        epoch_vlosses = []
        epoch_viouacc = []

        for i_batch, sample_batched in enumerate((t := tqdm(self.train_loader))):
          #print(i_batch+1, "/", len(self.train_loader), sample_batched['image'].size(), sample_batched['mask'].size())

          X = sample_batched['image'].float().to(self.device)
          Y = sample_batched['mask'].float().to(self.device)
          Y_idx = torch.argmax(Y, dim=1)  # NOTE: probably correct since it extracts indices i.e. the class the specific pixel belongs to

          # forward to net
          optim.zero_grad()
          out = self.model(X)
          #print(out)
          #print(Y_idx)
          #print(out.shape)
          #print(Y_idx.shape)
          # BUG: adding pixel_acc increases loss for some reason
          loss = loss_func(out, Y_idx)
          #pixel_acc = (torch.argmax(out, dim=1) == Y_idx).float().mean()
          #pixel_acc = (out == Y).float().mean()
          iou_acc = iou_coef(Y, out)
          self.writer.add_scalar('training running loss', loss.item(), l_idx)
          #self.writer.add_scalar('training running pixel accuracy', pixel_acc.item(), l_idx)
          self.writer.add_scalar('training running iou accuracy', iou_acc, l_idx)
          epoch_losses.append(loss.item())
          #epoch_pxaccuracies.append(pixel_acc.item())
          epoch_iouacc.append(iou_acc)
          loss.backward()
          optim.step()

          t.set_description("%d/%d: Batch TLoss: %.2f, IOU: %.2f"%(i_batch+1, len(self.train_loader), loss.item(), iou_acc))
          l_idx += 1
          #break # NOTE: this remains until we successfully overfit a single batch

        avg_epoch_loss = np.array(epoch_losses).mean()
        #avg_epoch_pxacc = np.array(epoch_pxaccuracies).mean()
        avg_epoch_iouacc = np.array(epoch_iouacc).mean()
        losses.append(avg_epoch_loss)
        #px_accuracies.append(avg_epoch_pxacc)
        iou_accuracies.append(avg_epoch_iouacc)
        print("[=>] Epoch average training loss: %.4f"%avg_epoch_loss)
        self.writer.add_scalar('training epoch avg loss', avg_epoch_loss, epoch)
        #print("[=>] Epoch average training pixel accuracy: %.4f"%avg_epoch_pxacc)
        #self.writer.add_scalar('training epoch avg loss', avg_epoch_pxacc, epoch)
        print("[=>] Epoch average training IOU accuracy: %.4f"%avg_epoch_iouacc)
        self.writer.add_scalar('training epoch avg IOU accuracy', avg_epoch_iouacc, epoch)

        # TODO: implement early stopping to avoid overfitting
        if self.early_stop:
          epoch_vlosses, epoch_viouacc = eval(epoch_vlosses, epoch_viouacc, train=True)
          avg_epoch_vloss = np.array(epoch_vlosses).mean()
          vlosses.append(avg_epoch_vloss)

    except KeyboardInterrupt:
      print("[~] Training stopped by user")
    print("[+] Training Done")
    model.save_model(self.model_path, self.model)

    # plot final training stats
    for idx, l in enumerate(losses):
      self.writer.add_scalar("final training loss", l, idx)
    #plt.plot(losses, label="train loss")

    if self.early_stop:
      #plt.plot(vlosses, label="val loss")
      pass

    for idx, acc in enumerate(iou_accuracies):
      self.writer.add_scalar("final training IOU accuracy", acc, idx)
    #plt.plot(iou_accuracies, label="train iou acc")
    #plt.show()

    if self.eval:
      val_losses = []
      val_iou = []
      val_losses, val_iou = eval(val_losses, val_iou)
      print("Average Evaluation Loss: %.4f"%(np.array(val_losses).mean()))
      print("Average Evaluation IOU: %.4f"%(np.array(val_iou).mean()))

    self.writer.close()
    return self.model
