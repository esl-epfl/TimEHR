import pickle

from typing import List

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd

# pd.options.mode.chained_assignment = None
# pd.options.mode.copy_on_write = False

from tqdm import tqdm


# Define your custom dataset class inheriting from torch.utils.data.Dataset
class ISTSDataset(Dataset):
    def __init__(
        self,
        static_data,
        temporal_data: List[np.ndarray],
        observation_times,
        outcome,
        dynamic_processor,
        static_processor,
        img_size=64,
        granularity=1,
        to_preprocess=False,
    ):

        self.img_size = img_size
        self.granularity = granularity  # granularity of the temporal data
        self.static_features = static_data.columns.tolist()
        self.static_data = static_data.values
        # self.static_data = static_data
        self.temporal_features = dynamic_processor["features"]
        # self.temporal_data = [x.values for x in temporal_data]
        self.temporal_data = temporal_data

        self.observation_times = observation_times
        self.outcome = outcome.values

        self.collate_fn = collate_fn

        # saving the dynamic and static processors
        self.dynamic_processor = dynamic_processor
        self.static_processor = static_processor

        self.to_preprocess = to_preprocess
        if to_preprocess:
            self.preprocess()
            self.collate_fn = collate_fn
        else:
            self.collate_fn = collate_fn

    def preprocess(self):

        temp = []
        for x, times in tqdm(
            zip(self.temporal_data, self.observation_times),
            desc="preprocessing temporal data into images",
            total=len(self.temporal_data),
        ):
            temp.append(
                pad2image(
                    x, times=times, img_size=self.img_size, granularity=self.granularity
                )
            )
        self.values = (
            torch.tensor([x[0] for x in temp]).unsqueeze(1).float()
        )  # [B,1,64,64
        self.masks = torch.tensor([x[1] for x in temp]).unsqueeze(1).float()
        self.static_data = torch.tensor(self.static_data)
        self.observation_times = self.observation_times
        self.outcome = torch.tensor(self.outcome)

        self.static_data = torch.cat([self.static_data, self.outcome], dim=1).float()

        return

    def __len__(self):
        return len(self.static_data)

    def __getitem__(self, idx):
        if self.to_preprocess:
            return (
                self.masks[idx],
                self.values[idx],
                self.static_data[idx],
                self.observation_times[idx],
                self.outcome[idx],
            )

        else:
            return (
                self.temporal_data[idx],
                self.static_data[idx],
                self.observation_times[idx],
                self.outcome[idx],
            )


def collate_fn(batch):
    masks = torch.concat([x[0] for x in batch]).unsqueeze(1)  # [B,1,64,64]
    values = torch.concat([x[1] for x in batch]).unsqueeze(1)  # [B,1,64,64]
    static_data = torch.stack([x[2] for x in batch])  # [B, D]
    observation_times = [x[3] for x in batch]
    outcome = torch.tensor([x[4] for x in batch])  # [B, 1]

    return (
        masks.float(),
        values.float(),
        static_data.float(),
        observation_times,
        outcome,
    )


def pad2image(temporal_data: np.ndarray, times: List = [], img_size=64, granularity=1):

    d_ts = temporal_data.shape[1]

    # if times are not provided, assume they are evenly spaced
    if len(times) == 0:
        times = np.arange(0, len(temporal_data) * granularity, granularity)

    assert len(temporal_data) == len(
        times
    ), "Temporal data and times should be same length"

    # check if times are devideable by granularity
    assert all(
        [x % granularity == 0 for x in times]
    ), "All times should be devideable by granularity"

    img_times = np.arange(0, img_size, granularity)  # [img_size]
    values = np.full((img_size, img_size), np.nan)

    mask_obaerved = [x in times for x in img_times]  # [img_size], True if observed
    values[mask_obaerved, :d_ts] = temporal_data  # [img_size, d_ts]

    masks = np.isnan(values).astype(int)
    masks = 1 - 2 * masks  # [0,1] -> [-1,1]. -1 for missing values
    values = np.nan_to_num(values, nan=0.0)
    # print(values.shape, masks.shape)
    # pad both dimensions to img_size

    # pad_col = img_size - values.shape[1] % img_size
    # # pad_row = img_size - values.shape[0] % img_size
    # values = np.pad(values, ((0, 0), (0, pad_col)), mode="constant", constant_values=0)
    # masks = np.pad(masks, ((0, 0), (0, pad_col)), mode="constant", constant_values=-1)
    # print(values.shape, masks.shape)

    return values, masks


def process_df(df_temporal, df_static, static_types):
    from sklearn.preprocessing import OneHotEncoder

    if not (df_static["id"].nunique() == df_temporal["id"].nunique()):
        # keep ids that are in both dataframes
        ids_in_static = df_static["id"].unique()
        ids_in_temporal = df_temporal["id"].unique()
        ids = np.intersect1d(ids_in_static, ids_in_temporal)
        df_static = df_static[df_static["id"].isin(ids)]
        df_temporal = df_temporal[df_temporal["id"].isin(ids)]

        # print warning and number of discarded ids
        print("Warning: ids are not the same in df_static and df_temporal")
        print(f"Discarded {len(ids_in_static)+len(ids_in_temporal)-2*len(ids)} ids")

    # check there is no missing data in df_static
    assert df_static.isnull().sum().sum() == 0, "There are missing values in df_static"

    temporal_features = df_temporal.columns.tolist()
    temporal_features.remove("id")
    temporal_features.remove("timepoint")
    len(temporal_features)

    stats = df_temporal[temporal_features].describe()
    dynamic_processor = dict()
    dynamic_processor["features"] = temporal_features
    dynamic_processor["mean"] = stats.loc["mean"].values
    dynamic_processor["std"] = stats.loc["std"].values
    # dynamic_processor['min'] = stats.loc['min'].values
    # dynamic_processor['max'] = stats.loc['max'].values

    # normalize
    df_temporal[temporal_features] = (
        df_temporal[temporal_features] - dynamic_processor["mean"]
    ) / dynamic_processor["std"]

    # remove outliers
    df_temporal[temporal_features] = df_temporal[temporal_features].apply(
        lambda x: x.mask(x.sub(x.mean()).div(x.std()).abs().gt(3))
    )

    dynamic_processor["min"] = df_temporal[temporal_features].min().values
    dynamic_processor["max"] = df_temporal[temporal_features].max().values

    # do min-max normalization # between -1 and 1
    df_temporal[temporal_features] = (
        df_temporal[temporal_features] - df_temporal[temporal_features].min()
    ) / (
        df_temporal[temporal_features].max() - df_temporal[temporal_features].min()
    ) * 2 - 1

    static_processor = dict()
    static_features = df_static.columns.tolist()
    static_features.remove("id")

    # df_static.isnull().sum()
    # df_static.describe()

    # standardize continuous variables
    static_processor = {
        "static_features": static_features,
        "static_types": static_types,
    }

    pd.options.mode.chained_assignment = None
    for var, type in zip(static_features, static_types):
        if type == "continuous":
            static_processor[var] = {
                "mean": df_static[var].mean(),
                "std": df_static[var].std(),
            }

            # standardize
            df_static.loc[:, var] = (
                df_static[var] - static_processor[var]["mean"]
            ) / static_processor[var]["std"]

        elif type == "categorical":
            ohe = OneHotEncoder()
            transformed = ohe.fit_transform(df_static[[var]])
            transformed
            mat_enc = pd.DataFrame.sparse.from_spmatrix(transformed).values.astype(int)
            new_cols = [var + str(i) for i in range(mat_enc.shape[1])]
            df_static.loc[:, new_cols] = mat_enc

            df_static = df_static.drop(columns=var)
        elif type == "binary":

            df_static[var] = df_static[var].astype(int)

    # for column in static_features:
    #     df_static[column] = df_static[column].fillna(df_static[column].mean())
    #     if column in ["Age", "Height", "Weight", "HospAdmTime"]:
    #         static_processor[column] = {
    #             "mean": df_static[column].mean(),
    #             "std": df_static[column].std(),
    #         }

    #         df_static[column] = (
    #             df_static[column] - df_static[column].mean()
    #         ) / df_static[column].std()

    # # discritze ICUType
    # if "ICUType" in static_features:

    #     ohe = OneHotEncoder()
    #     transformed = ohe.fit_transform(df_static[["ICUType"]])
    #     transformed
    #     mat_ICU_enc = pd.DataFrame.sparse.from_spmatrix(transformed).values.astype(int)
    #     new_cols = ["ICUType" + str(i) for i in range(mat_ICU_enc.shape[1])]
    #     df_static[new_cols] = mat_ICU_enc

    #     df_static = df_static.drop(columns="ICUType")

    # if "Gender" in static_features:
    #     df_static["Gender"] = df_static["Gender"].astype(int)

    if "id" in static_features:
        static_processor["features"].remove("id")
    static_processor["features_encoded"] = df_static.columns.tolist()
    static_processor["features_encoded"].remove("id")

    #########################

    # rearrange temporal data
    observation_data, temporal_dataframes = ([] for i in range(2))
    # temp_df_masks = []

    # sort by id
    df_static = df_static.sort_values(by="id")
    df_temporal = df_temporal.sort_values(by=["id", "timepoint"])

    # # check if both dataframes have the same ids
    # assert (
    #     df_static["id"].nunique() == df_temporal["id"].nunique()
    # ), "ids are not the same in df_static and df_temporal"

    # for id in tqdm(df_static["id"].unique(), desc="separating data"):
    #     temp_df = df_temporal[df_temporal["id"] == id]
    #     observations = temp_df["timepoint"].tolist()
    #     temp_df.set_index("timepoint", inplace=True)
    #     temp_df = temp_df.drop(columns=["id"])

    #     # # randomly select 10% of the dvalues matrix  to be missing
    #     # temp_df_mask = temp_df.notnull().astype(int)
    #     # temp_df_mask.columns = [f"masked_{col}" for col in temp_df_mask.columns]

    #     # add each to list
    #     observation_data.append(observations)
    #     temporal_dataframes.append(temp_df)
    #     # temp_df_masks.append(temp_df_mask)
    observation_data = df_temporal.groupby("id")["timepoint"].apply(list).tolist()

    temporal_dataframes = (
        df_temporal.groupby("id")[temporal_features]
        .apply(lambda x: x.values.tolist())
        .tolist()
    )
    temporal_dataframes = [np.array(x) for x in temporal_dataframes]

    outcome_data = df_static[["outcome"]].astype(int)  # only binary outcomes

    df_static.drop(columns=["id", "outcome"], inplace=True)

    return (
        temporal_dataframes,
        df_static,
        observation_data,
        outcome_data,
        dynamic_processor,
        static_processor,
    )


def get_datasets(config_data, split=0, preprocess=True):

    df_static = pd.read_csv(
        config_data.path_processed + f"/split{split}/df_static.csv"
    ).iloc[:]
    df_temporal = pd.read_csv(config_data.path_processed + f"/split{split}/df_ts.csv")
    train_ids = pickle.load(
        open(config_data.path_processed + f"/split{split}/train_ids.pkl", "rb")
    )

    # df_static.rename(columns={"RecordID": "id", "Label": "outcome"}, inplace=True)
    # df_temporal.rename(columns={"RecordID": "id", "Time": "timepoint"}, inplace=True)

    df_static_train = df_static[df_static.id.isin(train_ids)]  # .copy()
    df_temporal_train = df_temporal[df_temporal.id.isin(train_ids)]  # .copy()

    df_static_val = df_static[~df_static.id.isin(train_ids)]  # .copy()
    df_temporal_val = df_temporal[~df_temporal.id.isin(train_ids)]  # .copy()

    # creating train dataset
    (
        temporal_data,
        static_data,
        observation_times,
        outcome,
        dynamic_processor,
        static_processor,
    ) = process_df(
        df_temporal_train, df_static_train, static_types=config_data.static_types
    )

    train_dataset = ISTSDataset(
        static_data,
        temporal_data,
        observation_times,
        outcome,
        dynamic_processor,
        static_processor,
        img_size=config_data.img_size,
        granularity=config_data.granularity,
        to_preprocess=preprocess,
    )

    # creating val dataset
    (
        temporal_data,
        static_data,
        observation_times,
        outcome,
        dynamic_processor,
        static_processor,
    ) = process_df(
        df_temporal_val, df_static_val, static_types=config_data.static_types
    )

    val_dataset = ISTSDataset(
        static_data,
        temporal_data,
        observation_times,
        outcome,
        dynamic_processor,
        static_processor,
        img_size=config_data.img_size,
        granularity=config_data.granularity,
        to_preprocess=preprocess,
    )

    return train_dataset, val_dataset
