import pandas as pd
import numpy as np
import torchaudio
import os
import torch

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

class RainforestDataset(torch.utils.data.IterableDataset):  
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

    if self.caching_dir and not os.path.exists(self.caching_dir):
      os.mkdir(self.caching_dir)
    
  def encode_label(self, species_id, songtype_id):
    try:
      return self.labels.index((species_id, songtype_id))
    except:
      return len(self.labels)-1

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    data_start = min(self.csv_data.index)
    data_end = max(self.csv_data.index)

    if worker_info is None:  # single-process data loading, return the full iterator
      iter_start = data_start
      iter_end = data_end
    else:  # in a worker process
      # split workload
      per_worker = int(math.ceil((data_end - data_start) / float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = data_start + worker_id * per_worker
      iter_end = min(iter_start + per_worker, data_end)
        
    def iter_data():
      indices = list(range(iter_start, iter_end))
      if self.shuffle:
        np.random.shuffle(indices)
          
      for i in indices:
        if self.caching_dir and os.path.exists("%s/%s.pth" % (self.caching_dir, self.csv_data.iloc[i]["recording_id"])):
          spec = torch.load("%s/%s.pth" % (self.caching_dir, self.csv_data.iloc[i]["recording_id"]))
        else:
          snippet = torchaudio.load("%s/train/%s.flac" % (self.folder, self.csv_data.iloc[i]["recording_id"]), normalization=True)
          spec = self.make_spec(snippet[0])
          if self.caching_dir:
            torch.save(spec, "%s/%s.pth" % (self.caching_dir, self.csv_data.iloc[i]["recording_id"]))
        audio_len = snippet[0].shape[-1]
        sample_rate = snippet[1]
        spec_len = spec.shape[-1]
        step_size_seconds = audio_len/sample_rate/spec_len*self.output_size
        j = 1
        while j*self.output_size <= spec_len:
          in_obs_time = max((j-1)*step_size_seconds, self.csv_data.iloc[i]["t_min"]) < \
            min(j*step_size_seconds, self.csv_data.iloc[i]["t_max"])
          
          # Skip 50% of the non-observations
          if not in_obs_time and np.random.rand() < 0.5:
            continue
          
          label = self.encode_label(self.csv_data.loc[i]["species_id"], self.csv_data.loc[i]["songtype_id"]) if in_obs_time else self.encode_label(None, None)
          yield (spec[:,:,(j-1)*self.output_size:(j)*self.output_size], label)
          j += 1

    return iter_data()
