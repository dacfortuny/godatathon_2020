from torch.utils.data import Dataset, DataLoader
import torch


class NovartisDataset(Dataset):
    def __init__(self, volume_df):
        self.data = volume_df
        self.Xs = list()
        self.ys = list()

        self.volume_grouped = self.data.groupby(["country", "brand"])
        self.group_keys = list(self.volume_grouped.groups)

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, index):
        current_group = self.group_keys[index]
        df = self.volume_grouped.get_group(current_group)

        x = df[["volume"]][df["month_num"] < 0].values
        y = df[["volume"]][df["month_num"] >= 0].values

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y
