import torch
from torch.utils.data import Dataset


# dataloader class
class spikeTimeData(Dataset):
        """
        # data: 
        0 col: Time
        1 col: Neuron ID

        block_size:
        Transformer Window

        # dt
        Time intervals from data col 0

        # stoi, itos: dictionaries mapping neuron ID to transformer vocab
        and vice-versa.
        """

        def __init__(self, data, block_size, dt=None, stoi=None, itos=None):
                neurons = sorted(list(set(data)))
                if stoi is None:
                        data_size, population_size = len(data), len(neurons)
                else:
                        data_size, population_size = len(data), len(stoi)
                print('data has %d characters, %d unique.' % (data_size, population_size))
                
                if stoi and itos is not None:
                        self.stoi = stoi
                        self.itos = itos
                else:
                        self.stoi = { ch:i for i,ch in enumerate(neurons) }
                        self.itos = { i:ch for i,ch in enumerate(neurons) }
                
                self.block_size = block_size
                self.population_size = population_size
                self.data = data
                self.dt = dt

        def __len__(self):
                return len(self.data) - self.block_size

        def __getitem__(self, idx):
                # grab a chunk of (block_size + 1) characters from the data
                chunk = self.data[idx:idx + self.block_size + 1]
                # encode every character to an integer
                dix = [self.stoi[s] for s in chunk]

                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)
                
                # add temporal signal if it exists
                if self.dt is not None:
                    dt_chunk = self.dt[idx:idx + self.block_size + 1]
                    dt = torch.tensor(dt_chunk[:-1].to_numpy(), dtype=torch.float32)
                else:
                    dt = torch.zeros(x.size())
                
                x = torch.stack((x, dt), dim=1)
                # dty = torch.tensor(dt_chunk[1:].to_numpy(), dtype=torch.float32)
                return x, y