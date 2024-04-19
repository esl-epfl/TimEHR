import pickle

from torch.utils.data import Dataset, DataLoader


class Physio3(Dataset):
    def __init__(
        self,
        all_masks,
        all_values,
        all_sta,
        static_processor=None,
        dynamic_processor=None,
        transform=None,
        ids=None,
        max_len=None,
    ):
        self.num_samples = all_masks.shape[0]
        self.mask = all_masks
        self.value = all_values
        self.sta = all_sta
        self.transform = transform

        self.static_processor = static_processor
        self.dynamic_processor = dynamic_processor
        self.ids = ids
        self.max_len = max_len

        self.n_ts = len(self.dynamic_processor["mean"])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # image = self.data[idx]
        # label = self.labels[idx]

        # if self.transform:
        #     image = self.transform(image)

        return self.mask[idx], self.value[idx], self.sta[idx]


def load_data(path_train, path_eval, batch_size=64):

    with open(path_train, "rb") as file:
        train_dataset = pickle.load(file)
        train_schema = Physio3(
            train_dataset.mask,
            train_dataset.value,
            train_dataset.sta,
            static_processor=train_dataset.static_processor,
            dynamic_processor=train_dataset.dynamic_processor,
            ids=train_dataset.ids,
            max_len=train_dataset.max_len,
        )

    with open(path_eval, "rb") as file:
        val_dataset = pickle.load(file)
        val_schema = Physio3(
            val_dataset.mask,
            val_dataset.value,
            val_dataset.sta,
            static_processor=val_dataset.static_processor,
            dynamic_processor=val_dataset.dynamic_processor,
            ids=val_dataset.ids,
            max_len=val_dataset.max_len,
        )

    # train_loader = DataLoader(
    #         train_dataset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #     )

    # val_loader = DataLoader(
    #         val_dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #     )

    state_vars = list(train_dataset.dynamic_processor["mean"].keys())

    return train_dataset, val_dataset, train_schema, val_schema
