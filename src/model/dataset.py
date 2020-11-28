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

        # Split pre/post
        pre_idx = df["month_num"] < 0
        post_idx = df["month_num"] >= 0

        df_pre = df[pre_idx]
        df_post = df[post_idx]

        # Encoder (temporal) Features
        pre_vol_norm = df_pre[["volume_norm"]].values
        months = df_pre[["month_sin", "month_cos"]].values

        pre_vol_norm = torch.from_numpy(pre_vol_norm).float()
        months = torch.from_numpy(months).float()

        encoder_temp_features = torch.cat([pre_vol_norm, months], dim=1)

        # Encoder numerical features
        encoder_num_features = [
            df["num_generics"].unique().item(),
            df["channel_rate_A"].unique().item(),
            df["channel_rate_B"].unique().item(),
            df["channel_rate_C"].unique().item()
        ]
        encoder_num_features = torch.tensor(encoder_num_features,
                                            dtype=torch.float)

        # Encoder categorical features
        encoder_cat_features = [
            df["country_id"].unique().item(),
            df["brand_id"].unique().item(),
            df["package_id"].unique().item(),
            df["therapeutic_id"].unique().item()
        ]

        encoder_cat_features = torch.tensor(encoder_cat_features,
                                            dtype=torch.long)

        # Decoder (temporal) Features
        pre_vol_norm = df_post[["volume_norm"]].values
        months = df_post[["month_sin", "month_cos"]].values

        pre_vol_norm = torch.from_numpy(pre_vol_norm).float()
        months = torch.from_numpy(months).float()

        decoder_temp_features = torch.cat([pre_vol_norm, months], dim=1)

        # Y
        y_norm = df_post[["volume_norm"]].values
        y_norm = torch.from_numpy(y_norm).float()

        return {
            "encoder_temp_features": encoder_temp_features,
            "encoder_num_features": encoder_num_features,
            "encoder_cat_features": encoder_cat_features,
            "decoder_temp_features": decoder_temp_features,
            "y_norm": y_norm,
            "avg_12_volume": avg_12_volume,
            "max_volume": max_volume
        }
