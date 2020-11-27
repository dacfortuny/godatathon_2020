from torch.utils.data import Dataset, DataLoader


class NovartisDataset(Dataset):
    def __init__(self, volume_df):
        self.data = volume_df
        self.Xs = list()
        self.ys = list()

        volume_grouped = self.data.groupby(["country", "brand"])
        for _, df in volume_grouped:
            self.Xs.append(df[["volume"]][df["month_num"] < 0].values)
            self.ys.append(df[["volume"]][df["month_num"] >= 0].values)

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, index):
        return self.Xs[index], self.ys[index]
