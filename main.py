
import numpy as np
import pandas as pd 
import torch
import math
import torchaudio
import torchvision
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_data(file):
  data = pd.read_csv(file)
  labels = [(data.loc[i]["species_id"], data.loc[i]["songtype_id"]) for i in data.index]
  labels = list(dict.fromkeys(labels))
  labels.sort()
  labels.append((None, None))

  msk = np.random.rand(len(data)) < 0.8
  data_train = data[msk]
  data_test = data[~msk]

  return data_train, data_test, labels

class BirdDataset(torch.utils.data.IterableDataset):  
  def transform(self, x):
    return torch.stack([
      torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=6000, hop_length=1200, f_min=180, f_max=14000, n_mels=self.output_size, pad=2, power=0.25)(x)[0],
      torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=6000, hop_length=1200, f_min=180, f_max=14000, n_mels=self.output_size, pad=2, power=2)(x).log()[0],
      torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=6000, hop_length=1200, f_min=180, f_max=14000, n_mels=self.output_size, pad=2, power=6)(x).log()[0]])
  
  def __init__(self, dataframe, labels):
    self.folder = "/kaggle/input/rfcx-species-audio-detection"
    self.caching_dir = "/tmp/rfcx-species-audio-detection"
    self.csv_data = dataframe
    self.labels = labels
    self.output_size = 224
    self.shuffle = True
    self.make_spec = lambda x: self.transform(x)
    
  def encode_label(self, species_id, songtype_id):
    try:
      return self.labels.index((species_id, songtype_id))
    except:
      return len(self.labels)-1

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # single-process data loading, return the full iterator
      iter_start = self.csv_data.index.start
      iter_end = self.csv_data.index.stop
    else:  # in a worker process
      # split workload
      per_worker = int(math.ceil((self.csv_data.index.stop - self.csv_data.index.start) / float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = self.csv_data.index.start + worker_id * per_worker
      iter_end = min(iter_start + per_worker, self.csv_data.index.stop)
        
    def iter_data():
      indices = list(range(iter_start, iter_end))
      if self.shuffle:
        np.random.shuffle(indices)
          
      for i in indices:
        if self.caching_dir and os.path.exists("%s/%s.pth" % (self.caching_dir, self.csv_data.loc[i]["recording_id"])):
          spec = torch.load("%s/%s.pth" % (self.caching_dir, self.csv_data.loc[i]["recording_id"]))
        else:
          snippet = torchaudio.load("%s/train/%s.flac" % (self.folder, self.csv_data.loc[i]["recording_id"]), normalization=True)
          spec = self.make_spec(snippet[0])
          if self.caching_dir:
            torch.save(spec, "%s/%s.pth" % (self.caching_dir, self.csv_data.loc[i]["recording_id"]))
        audio_len = snippet[0].shape[-1]
        sample_rate = snippet[1]
        spec_len = spec.shape[-1]
        step_size_seconds = audio_len/sample_rate/spec_len*self.output_size
        j = 1
        while j*self.output_size <= spec_len:
          in_obs_time = max((j-1)*step_size_seconds, self.csv_data.loc[i]["t_min"]) < \
            min(j*step_size_seconds, self.csv_data.loc[1]["t_max"])
          
          # Skip 50% of the non-observations
          if not in_obs_time and np.random.rand() < 0.5:
            continue
          
          label = self.encode_label(self.csv_data.loc[i]["species_id"], self.csv_data.loc[i]["songtype_id"]) if in_obs_time else self.encode_label(None, None)
          yield (spec[:,:,(j-1)*self.output_size:(j)*self.output_size], label)
          j += 1

    return iter_data()


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
  data_train, data_test, labels = read_data("rfcx-species-audio-detection/train_tp.csv")

  dataloader = torch.utils.data.DataLoader(BirdDataset(data_train, labels), batch_size=64, num_workers=8, pin_memory=True)
  dataloader_test = torch.utils.data.DataLoader(BirdDataset(data_test, labels), batch_size=64, num_workers=8, pin_memory=True)

  model = torchvision.models.resnet18(pretrained=True)

  num_ftrs = model.fc.in_features
  model.fc = torch.nn.Linear(num_ftrs, len(labels))
  model = model.to(device)

  optim = torch.optim.Adam(model.parameters(), lr=1e-4)
  criterion = torch.nn.CrossEntropyLoss()

  for epoch in range(10):
    train_epoch(dataloader, model, optim, criterion)
    validate(dataloader_test, model, criterion)

main()