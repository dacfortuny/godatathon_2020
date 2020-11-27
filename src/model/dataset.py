from torch.utils.data import Dataset, DataLoader
import torch


class NovartisDataset(Dataset):
    def __init__(self, volume_df):
        self.data = volume_df
        self.Xs = list()
        self.ys = list()

        volume_grouped = self.data.groupby(["country", "brand"])

        self.group_keys = list(volume_grouped.groups.keys())

        for _, df in volume_grouped:
            self.Xs.append(df[["volume"]][df["month_num"] < 0].values)
            self.ys.append(df[["volume"]][df["month_num"] >= 0].values)

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, index):
        x = torch.from_numpy(self.Xs[index]).float()
        y = torch.from_numpy(self.ys[index]).float()
        return x, y
