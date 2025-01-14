import torch
from torch.utils.data import Dataset

class WindowedFrameDisDataLoader(torch.utils.data.Dataset):
    def __init__(self, data, target, domain, idx, transform=None):
        self.data = torch.from_numpy(data).float()
        # self.data = self.data.permute(0, 2, 1)  #  set it to batch_num, channel, time_dim
        self.idx = torch.from_numpy(idx)
        self.domain = torch.from_numpy(domain).long()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        i = self.idx[index]
        d = self.domain[index]
        if self.transform:
            x = self.transform(x)
        return x, y, d, i

    def __len__(self):
        return len(self.data)