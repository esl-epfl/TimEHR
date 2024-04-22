from omegaconf import DictConfig, OmegaConf

import os
import wandb
from dotenv import load_dotenv

# python eval.py Results/p12-s0

import argparse
import yaml

from Utils.models import TimEHR

from Utils.utils import mat2df

from Utils.evaluation import evaluate

from data.data_utils import get_datasets

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate the results of the model")
    parser.add_argument("path", type=str, help="Path to the results folder")
    parser.add_argument(
        "--method", type=str, default="ctgan", help="method for generating static data"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of synthetic samples to generate, if 0, generate as many as the samples in training data",
    )
    args = parser.parse_args()

    print(args.path)

    # setup wandb
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_KEY"))

    # load progress.yaml
    progress = OmegaConf.load(args.path + "/progress.yaml")

    # load config.yaml
    config = OmegaConf.load(args.path + "/config.yaml")

    # loading data
    # loading data
    train_dataset, val_dataset = get_datasets(
        config.data, split=config.split, preprocess=True
    )

    # initialize the model
    model = TimEHR(config)

    # update the model from the saved weights
    model.from_pretrained(
        path_cwgan=progress.cwgan_run_path, path_pix2pix=progress.pix2pix_run_path
    )

    # get synthetic data
    counts = args.count if args.count != 0 else len(train_dataset)
    fake_static, fake_data = model.generate(
        train_dataset, count=counts, method=args.method
    )
    df_ts_fake, df_static_fake = mat2df(
        fake_data,
        fake_static,
        train_dataset.dynamic_processor,
        train_dataset.static_processor,
    )

    # get train data
    train_static, train_data = model._get_data(train_dataset)
    df_ts_train, df_static_train = mat2df(
        train_data,
        train_static,
        train_dataset.dynamic_processor,
        train_dataset.static_processor,
    )

    # get test data
    val_static, val_data = model._get_data(val_dataset)
    df_ts_test, df_static_test = mat2df(
        val_data,
        val_static,
        val_dataset.dynamic_processor,
        val_dataset.static_processor,
    )

    # prepare inputs
    inputs = {
        # normalized data
        "fake_static": fake_static,
        "fake_data": fake_data,
        "train_static": train_static,
        "train_data": train_data,
        "test_static": val_static,
        "test_data": val_data,
        # dataframes from normalized data
        "df_ts_fake": df_ts_fake,
        "df_static_fake": df_static_fake,
        "df_ts_train": df_ts_train,
        "df_static_train": df_static_train,
        "df_ts_test": df_ts_test,
        "df_static_test": df_static_test,
        "state_vars": train_dataset.temporal_features,
    }

    # we save the results in a wandb task
    wandb_task_name = f"{config.data.name}-s{config.split}"
    wandb_config = {"data": config.data.name, "split": config.split}
    evaluate(inputs, wandb_task_name=wandb_task_name, config=wandb_config)
