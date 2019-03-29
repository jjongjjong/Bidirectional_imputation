from torch.utils.data import Dataset
import torch
import pandas as pd


class activityDataset(Dataset):
    def __init__(self, root_dir,folder,mode , normalize=None):
        self.dataset = torch.from_numpy(pd.read_csv('{}{}/{}.csv'.format(root_dir,folder,mode), index_col=0).values)
        print(self.dataset.shape)
        self.corr_data = self.dataset[:, :720]
        self.raw_data = self.dataset[:, 720:720 * 2]
        self.mask_data = self.dataset[:, 720 * 2:720 * 3]
        self.missing_rate =self.dataset[:, -14:-13]
        self.min_max = self.dataset[:, -13:-11]
        self.mean_var = self.dataset[:, -11:-9]
        self.total_mean_var = self.dataset[:, -2:]
        self.bmi = self.dataset[:, -8:-7]
        self.sex = self.dataset[:, -7:-6]
        self.age = self.dataset[:, -6:-5]
        self.race = self.dataset[:, -5:-4]
        self.week = self.dataset[:, -3:-2]
        self.norm_data = self.dataset[:, 720:720 * 2]

        if normalize == 'zero':
            self.norm_data = self.zero_normalize()
        elif normalize == 'minmax':
            self.norm_data = self.min_max_normalize()
        elif normalize == 'total_zero':
            self.norm_data = self.batch_zero_normalize()

    def __len__(self):
        return (self.dataset.shape[0])

    def __getitem__(self, idx):
        return self.corr_data[idx], self.raw_data[idx], self.mask_data[idx], self.norm_data[idx], \
               {'mean_var': self.mean_var[idx], 'min_max': self.min_max[idx],
                'total_mean_var': self.total_mean_var[idx]}, \
               {'bmi': self.bmi[idx], 'sex': self.sex[idx], 'age': self.age[idx], 'race': self.race[idx],
                'week': self.week[idx]}

    def zero_normalize(self):
        mean = self.mean_var[:, 0].view(-1, 1)
        std = torch.sqrt(self.mean_var[:, 1].view(-1, 1))
        norm = (self.raw_data - mean) / std
        #norm[self.corr_data == -1] = 0
        return norm

    def min_max_normalize(self):
        min = self.min_max[:, 0].view(-1, 1)
        max = self.min_max[:, 1].view(-1, 1)
        norm = (self.raw_data - min) / (max - min)
        #norm[self.corr_data == -1] = 0
        return norm

    def batch_zero_normalize(self):
        mean = self.total_mean_var[:, 0].view(-1, 1)
        std = torch.sqrt(self.total_mean_var[:, 1].view(-1, 1))
        norm = (self.raw_data - mean) / std
        #norm[self.corr_data == -1] = 0
        return norm

