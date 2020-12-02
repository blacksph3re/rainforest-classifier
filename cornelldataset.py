import pandas as pd
import numpy as np
import torchaudio
import os
import torch


def transform(output_size, x):
     return torch.stack([
            torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=6000, hop_length=1200, f_min=180, f_max=14000, n_mels=output_size, pad=2, power=0.25)(x)[0],
            torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=6000, hop_length=1200, f_min=180, f_max=14000, n_mels=output_size, pad=2, power=2)(x).log()[0],
            torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=6000, hop_length=1200, f_min=180, f_max=14000, n_mels=output_size, pad=2, power=6)(x).log()[0]])

class CornellBirdDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataframe, labels):
        self.folder = "/"
        self.caching_dir = "/tmp/birdsong"
        self.csv_data = dataframe
        self.labels = labels
        self.output_size = 224
        self.shuffle = True
        self.make_spec = lambda x: transform(self.output_size, x)
        
        if self.caching_dir and not os.path.exists(self.caching_dir):
            os.mkdir(self.caching_dir)
      
    def encode_label(self, species):
        try:
            return self.labels.index(species)
        except:
            return len(self.labels)-1
    
    def __iter__(self):
        data_start = min(self.csv_data.index)
        data_stop = max(self.csv_data.index)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = data_start
            iter_end = sdata_stop
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((data_stop - data_start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = data_start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, data_stop)
            
        def iter_data():
            indices = list(range(iter_start, iter_end))
            if self.shuffle:
                np.random.shuffle(indices)
                
            for i in indices:
                if self.caching_dir and os.path.exists("%s/%s.pth" % (self.caching_dir, self.csv_data.iloc[i]["filename"])):
                    spec = torch.load("%s/%s.pth" % (self.caching_dir, self.csv_data.iloc[i]["filename"]))
                else:
                    snippet = torchaudio.load(self.csv_data.iloc[i]["filename_full"], normalization=True)[0]
                    if snippet[1] != 48000:
                        snippet[0] = torchaudio.transforms.Resample(orig_freq=snippet[1], new_freq=48000)(snippet[0])
                    spec = self.make_spec(snippet[0])
                    if self.caching_dir:
                        torch.save(spec, "%s/%s.pth" % (self.caching_dir, self.csv_data.iloc[i]["filename"]))
                
                spec_len = spec.shape[-1]
                pos = int(np.random.uniform(0, self.output_size/4))
                label = self.encode_label(self.csv_data.iloc[i]["species"])

                while pos < spec_len-self.output_size:
                                     
                    yield (spec[:,:,pos:pos+self.output_size], label)
                    pos += int(np.random.uniform(self.output_size/4, self.output_size))

        return iter_data()


def load_cornell_data(folder):

  data = pd.read_csv("%s/birdsong-recognition/train.csv" % folder)
  data_ext1 = pd.read_csv("%s/xeno-canto-bird-recordings-extended-a-m/train_extended.csv" % folder)
  data_ext2 = pd.read_csv("%s/xeno-canto-bird-recordings-extended-n-z/train_extended.csv" % folder)

  data["filename_full"] = ("%s/birdsong-recognition/train_audio/" % folder) + data["ebird_code"] + "/" + data["filename"]
  data_ext1["filename_full"] = ("%s/xeno-canto-bird-recordings-extended-a-m/A-M/" % folder) + data_ext1["ebird_code"] + "/" + data_ext1["filename"]
  data_ext2["filename_full"] = ("%s/xeno-canto-bird-recordings-extended-n-z/N-Z/" % folder) + data_ext2["ebird_code"] + "/" + data_ext2["filename"]

  data = pd.concat([data, data_ext1, data_ext2]).reset_index(drop=True)

  labels = list(dict.fromkeys(data["species"]))
  labels.sort()
  labels.append(None)

  msk = np.random.rand(len(data)) < 0.8 
  data_train = data[msk].reset_index(drop=True)
  data_test = data[~msk].reset_index(drop=True)

  return data_train, data_test, labels
