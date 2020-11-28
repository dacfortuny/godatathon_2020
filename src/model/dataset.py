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

        # Average/max volume of prev 12 months (used for metric)
        avg_12_volume = df["avg_12_volume"].unique().item()
        max_volume = df["max_volume"].unique().item()

        x_norm = df[["volume_norm"]][df["month_num"] < 0].values
        y_norm = df[["volume_norm"]][df["month_num"] >= 0].values

        x_norm = torch.from_numpy(x_norm).float()
        y_norm = torch.from_numpy(y_norm).float()

        return {"x_norm": x_norm,
                "y_norm": y_norm,
                "avg_12_volume": avg_12_volume,
                "max_volume": max_volume}
