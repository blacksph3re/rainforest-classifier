
import numpy as np
import pandas as pd 
import torch
import math
import torchaudio
import torchvision
import matplotlib.pyplot as plt
from .cornelldataset import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_epoch(dataloader, model, optim, criterion):
  loss_sum = 0
  steps = 0
  for inputs, targets in dataloader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    optim.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optim.step()

    loss_sum += loss.detach().cpu()
    steps += 1
  print("Train loss: %f" % (loss_sum/steps))

def validate(dataloader, model, criterion):
  correct_preds = 0
  samples = 0
  loss_sum = 0
  steps = 0

  with torch.no_grad():
    for inputs, targets in dataloader:
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)
      pred = torch.argmax(outputs, dim=1)
      correct_preds += torch.sum(pred == targets).cpu().item()
      loss_sum += criterion(outputs, targets).cpu().item()
      samples += len(inputs)
      steps += 1
  
  print("Validation loss: %f, accuracy: %f" % (loss_sum/steps, correct_preds/samples))


def main():
  # data_train, data_test, labels = read_data("rfcx-species-audio-detection/train_tp.csv")

  # dataloader = torch.utils.data.DataLoader(RainforestDataset(data_train, labels), batch_size=64, num_workers=8, pin_memory=True)
  # dataloader_test = torch.utils.data.DataLoader(RainforestDataset(data_test, labels), batch_size=64, num_workers=8, pin_memory=True)

  data_train, data_test, labels = read_cornell_data("cornell-bird-dataset")


  model = torchvision.models.resnet18(pretrained=True)

  num_ftrs = model.fc.in_features
  model.fc = torch.nn.Linear(num_ftrs, len(labels))
  model = model.to(device)

  optim = torch.optim.Adam(model.parameters(), lr=1e-4)
  criterion = torch.nn.CrossEntropyLoss()

  for epoch in range(100):
    train_epoch(dataloader, model, optim, criterion)
    validate(dataloader_test, model, criterion)
    if epoch % 5 == 0:
      torch.save(model, "model-%d.pt" % epoch)

main()