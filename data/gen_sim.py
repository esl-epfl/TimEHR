# %% [markdown]
# # Libraries


# from Utils.data import Physio3

# general
import numpy as np
import pandas as pd
import argparse

# import custom libraries
import sys
import os
import tqdm
import pickle
import yaml

# %%

# plotly
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


# %%
# folder paths

PATH_RAW = "./raw/"
PATH_PROCESSED = "./processed/"
PATH_YAML = "../configs/data/"
# create folder if not exists
os.makedirs(PATH_RAW, exist_ok=True)
os.makedirs(PATH_PROCESSED, exist_ok=True)


# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.preprocessing import OneHotEncoder

# %%
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def generate_poisson_binary_vector(length, lambda_value):
    # Generate random counts from a Poisson distribution
    poisson_counts = np.random.poisson(lambda_value, length)

    # Convert counts to a binary vector
    binary_vector = np.where(poisson_counts > 0, 1, 0)

    return binary_vector


def main():
    global DATASET
    for n_vars in opt.n_vars:
        for lambda_value in opt.lambdas:
            # %%
            # Set random seed for reproducibility
            np.random.seed(42)

            # Parameters
            # n_vars = 25
            # lambda_value = 2
            n_samples = 10000
            n_timestamps = 128
            period_min = 16
            period_max = 48
            TARGET_DIM = 128 if n_vars == 128 else 64
            # TARGET_DIM = n_vars
            GRAN = 1

            # Generate sinusoidal data with random phases
            data = np.zeros((n_samples, n_timestamps, n_vars))
            periods = np.random.randint(period_min, period_max + 1, n_vars)

            for var in range(n_vars):
                phase = np.random.uniform(0, 2 * np.pi, 1)
                for sample in range(n_samples):
                    t = np.arange(n_timestamps)
                    amplitude = np.random.uniform(0.9, 1, 1)
                    noise = np.random.normal(0, 0.1, n_timestamps)
                    baseline = np.random.uniform(-0.5, 0.5, 1)
                    noise_period = np.random.randint(1, 5)
                    noise_phase = np.random.uniform(0, 2 * np.pi / 50, 1)
                    data[sample, :, var] = (
                        baseline
                        + amplitude
                        * np.sin(2 * np.pi * t / (periods[var] + noise_period) + phase)
                        + noise
                    )

            # Simulate random missing values based on a Poisson process
            missing_mask = (
                np.array(
                    [
                        generate_poisson_binary_vector(n_timestamps, lambda_value)
                        for i in range(n_samples * n_vars)
                    ]
                )
                .reshape(n_samples, n_vars, n_timestamps)
                .transpose(0, 2, 1)
            )

            mr = (missing_mask == 0).sum() / missing_mask.size * 100
            print(f"missing rate: {mr:.2f}%")

            # apply missing mask
            data_with_missing = np.where(missing_mask == 0, np.nan, data)

            # Create a plot using Plotly
            # You may need to install plotly via pip if you haven't already: pip install plotly
            sample_idx = 0  # Choose a sample to plot
            fig = go.Figure()

            for var in range(5):
                _ = fig.add_trace(
                    go.Scatter(
                        x=np.arange(n_timestamps),
                        y=data_with_missing[sample_idx, :, var],
                        mode="lines",
                        name=f"Variable {var+1}",
                    )
                )

            _ = fig.update_layout(
                title="Simulated Multivariate Time Series with Missing Data",
                xaxis_title="Time Steps",
                yaxis_title="Value",
            )
            # fig.show()

            data.shape
            data_with_missing.shape

            # %%
            # convert to dataframe

            col_names = [f"var_{i}" for i in range(n_vars)]

            df = pd.DataFrame(data_with_missing.reshape(-1, n_vars), columns=col_names)

            df.to_csv(PATH_RAW + f"sim-l{lambda_value}-d{n_vars}.csv", index=False)

            yaml_dict = {
                "name": DATASET,
                "path_raw": f"./data/raw",
                "path_processed": f"./data/processed/sim-l{lambda_value}-d{n_vars}",
                "img_size": TARGET_DIM,
                "granularity": 1,
                "n_split": 1,
                "seed": 42,
            }

            with open(PATH_YAML + f"sim-l{lambda_value}-d{n_vars}.yaml", "w") as file:
                yaml.dump(yaml_dict, file)


if __name__ == "__main__":

    DATASET = "sim"
    parser = argparse.ArgumentParser(description="Data Preprocessing")

    # # parser.add_argument('--dataset', type=str, default='p12', help='dataset', dest="DATASET" )
    # parser.add_argument('--d', type=int, default=32, help='number of variables', dest="n_vars")
    # parser.add_argument('--lambda', type=float, default=0.5, help='number of variables', dest="lambda_value")
    parser.add_argument(
        "--n-vars",
        nargs="+",
        help="A list of integers.",
        type=int,
        default=[32],
        dest="n_vars",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        help="A list of floats.",
        type=float,
        default=[0.5],
        dest="lambdas",
    )

    opt = parser.parse_args()

    print(opt.n_vars)
    print(opt.lambdas)

    # lambda_value = opt.lambda_value
    # n_vars = opt.n_vars

    # DATASET = f'sim-l{lambda_value}-d{n_vars}'
    # SUFFLE_VARS = False
    pass
    main()
