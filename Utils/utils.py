# stdlib
import os
from typing import Any, Dict, List, Tuple

# third party
import numpy as np
import pandas as pd

# import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torchvision
import wandb
from plotly.subplots import make_subplots


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    y: torch.Tensor = None,
    device: str = "cpu",
) -> torch.Tensor:
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)  # bs,2,64,64

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, y=y)  # bs,1,1,1

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def compute_corr_mat(
    all_values: torch.Tensor,
    all_masks: torch.Tensor,
    NVARS: int = 40,
    method: str = "none",
) -> np.ndarray:
    # all_values.shape = (*, n_vars) in [-1,1]
    # all_masks.shape = (*, n_vars) -1: missing, 1: not missing
    # method = 'none' or 'ffill'

    IMG_HEIGHT = all_values.shape[-2]
    NUM_SAMPLES = all_values.shape[0]

    # define a dataframe
    df = pd.DataFrame(all_values[:, :, :, :NVARS].reshape(-1, NVARS))
    df[(all_masks[:, :, :, :NVARS] < 0).reshape(-1, NVARS)] = np.nan

    if method == "ffill":
        # forward fill with limit of 6
        df["id"] = np.repeat(np.arange(NUM_SAMPLES), IMG_HEIGHT)
        # 0 0 0 0 ... 0 (L) | ....| 100 100 100 (L)
        df = df.groupby("id").ffill(limit=6)

    corr_mat = df.corr().values
    corr_mat[np.abs(corr_mat) < 0.2] = 0
    corr_mat[np.triu_indices_from(corr_mat)] = 0

    return corr_mat


def mat2img(mat: torch.Tensor) -> np.ndarray:
    X = mat.cpu().detach().numpy()

    # def grayscale_to_blue_red(x):
    #     if x < 0.5:
    #         return (int(255 * (1 - x)), int(255 * x), int(255 * x))
    #     else:
    #         return (int(255 * (1 - x)), int(255 * (1 - x)), int(255 * x))

    red_values = (255 * (1 - X[0])).astype(int)  # shape (H, W)
    blue_values = (255 * X[0]).astype(int)  # shape (H, W)
    green_values = (255 * X[0]).astype(int)  # shape (H, W)
    green_values[X[0] >= 0.5] = (255 * (1 - X[0, X[0] >= 0.5])).astype(int)
    # Apply the color mapping to the grayscale image

    # colored_image = np.apply_along_axis(grayscale_to_blue_red, 0, X[[0],:,:])
    colored_image = np.stack(
        [red_values, green_values, blue_values], axis=0
    )  # shape (3, H, W)

    colored_image = np.uint8(colored_image)
    colored_image[:, X[1, :, :] < 0.5] = 255

    return colored_image


def save_examples(
    real: torch.Tensor, fake: torch.Tensor, n_ts: int = 40, epoch_no: int = -1
) -> None:
    # everythinig is between -1 and 1

    NVARS = n_ts  # number of time series variables
    L_CUT = NVARS + 1
    N_samples = 9

    images = []

    for i in range(N_samples):

        img_grid_real = torchvision.utils.make_grid(
            real[i, :, :, :L_CUT], normalize=True, nrow=1
        )  # .unsqueeze(1) # shape (n_channels, H, W)
        img_grid_fake = torchvision.utils.make_grid(
            fake[i, :, :, :L_CUT], normalize=True, nrow=1
        )  # .unsqueeze(1) # shape (n_channels, H, W)

        upscaled_img_grid_real = torch.nn.functional.interpolate(
            img_grid_real.unsqueeze(0), scale_factor=4, mode="nearest"
        ).squeeze(
            0
        )  # shape (n_channels, 4H, 4W)
        upscaled_img_grid_fake = torch.nn.functional.interpolate(
            img_grid_fake.unsqueeze(0), scale_factor=4, mode="nearest"
        ).squeeze(
            0
        )  # shape (n_channels, 4H, 4W)

        real_image = mat2img(upscaled_img_grid_real)  # [3, 4H, 4W]
        fake_image = mat2img(upscaled_img_grid_fake)  # [3, 4H, 4W]

        # for the fake_image, set the white pixels to #fadf93
        def set_white_to_gray(image_tensor: np.ndarray) -> np.ndarray:
            # Check where all three channels are 255 (white)
            white_mask = np.all(image_tensor == 255, axis=0)

            # Set these locations to gray (128, 128, 128)
            # Update the Red, Green, and Blue channels 252, 238, 197
            image_tensor[0, white_mask] = 252  # Red
            image_tensor[1, white_mask] = 238  # Green
            image_tensor[2, white_mask] = 197  # Blue

            return image_tensor

        fake_image = set_white_to_gray(fake_image)

        # add a watermark to fake image with gray
        def add_watermark(
            image_tensor: np.ndarray,
            text: str = "Watermark",
            position: Any = "center",
            font_size: int = 40,
            font_color: Tuple = (128, 128, 128),
        ) -> np.ndarray:
            # third party
            from PIL import Image, ImageDraw, ImageFont

            # Convert the tensor to a PIL image
            image = Image.fromarray(
                np.transpose(image_tensor, (1, 2, 0)).astype("uint8")
            )

            # Create an ImageDraw object
            draw = ImageDraw.Draw(image)

            # Load a font
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            # Calculate the text position for center alignment using textbbox
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if position == "center":
                W, H = image.size
                x = (W - text_width) / 2
                y = (H - text_height) / 2
            else:
                x, y = position

            # Add text to image
            draw.text((x, y), text, font=font, fill=font_color)

            # Convert back to a numpy array if needed
            result_tensor = np.array(image).transpose(2, 0, 1)
            return result_tensor

        fake_image = add_watermark(fake_image, text="Synthetic")

        # add a black border on top, bottom and left of real image
        real_image = np.pad(
            real_image, ((0, 0), (2, 2), (2, 0)), "constant", constant_values=0
        )
        # add a gray border on right of real image

        real_image = np.pad(
            real_image, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=128
        )

        # add a black border on top, bottom and right of fake image
        fake_image = np.pad(
            fake_image, ((0, 0), (2, 2), (0, 2)), "constant", constant_values=0
        )
        # add a gray border on left of fake image
        fake_image = np.pad(
            fake_image, ((0, 0), (0, 0), (1, 0)), "constant", constant_values=196
        )

        real_fake_image = np.concatenate(
            [real_image, fake_image], axis=2
        )  # [3, 4H, 8W]

        images.append(real_fake_image)

    # we have 9 real images and 9 fake images
    # we want to make a 3x6 grid
    # we want to alternate real and fake images:
    # example first row: real1, fake1, real2, fake2, real3, fake3

    tot_H = images[0].shape[1] * 3
    tot_W = images[0].shape[2] * 3

    X = np.concatenate(images, axis=2).reshape(3, tot_H, tot_W)
    X = np.concatenate(
        [
            np.concatenate([images[0], images[1], images[2]], axis=2),
            np.concatenate([images[3], images[4], images[5]], axis=2),
            np.concatenate([images[6], images[7], images[8]], axis=2),
        ],
        axis=1,
    )

    # # map to RGB
    # third party
    from PIL import Image

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(X.transpose(1, 2, 0))

    # pil_image.save("output_image.png")

    if wandb.run is not None:
        if epoch_no == -1:  # during evaluation
            wandb.log(
                {
                    "Real-vs-Synthetic": [
                        wandb.Image(pil_image, caption="Real vs Synthetic")
                    ]
                }
            )
        else:  # when called during training
            wandb.log(
                {
                    "Real-vs-Synthetic": [
                        wandb.Image(pil_image, caption="Real vs Synthetic")
                    ]
                },
                step=epoch_no,
                commit=False,
            )

    return pil_image


def save_examples_dopp(
    real: torch.Tensor, fake: torch.Tensor, NVARS: int = 35, epoch: int = 0
) -> None:
    # everythinig is between -1 and 1
    # NVARS = opt.NVARS
    L_CUT = NVARS + 1
    N_samples = 9

    images = []

    # create a subplot
    fig = make_subplots(rows=1, cols=2)
    for i in range(5):
        _ = fig.add_trace(
            go.Scatter(x=np.arange(64), y=real[0, 0, :, i], name="real"), row=1, col=1
        )
        _ = fig.add_trace(
            go.Scatter(x=np.arange(64), y=fake[0, 0, :, i], name="fake"), row=1, col=2
        )

    wandb.log({"Real-vs-Fake2": wandb.Plotly(fig)}, step=epoch, commit=False)

    for i in range(N_samples):

        img_grid_real = torchvision.utils.make_grid(
            real[i, :, :, :L_CUT], normalize=True, nrow=1
        )  # .unsqueeze(1) # shape (n_channels, H, W)
        img_grid_fake = torchvision.utils.make_grid(
            fake[i, :, :, :L_CUT], normalize=True, nrow=1
        )  # .unsqueeze(1) # shape (n_channels, H, W)

        upscaled_img_grid_real = torch.nn.functional.interpolate(
            img_grid_real.unsqueeze(0), scale_factor=4, mode="nearest"
        ).squeeze(
            0
        )  # shape (n_channels, 4H, 4W)
        upscaled_img_grid_fake = torch.nn.functional.interpolate(
            img_grid_fake.unsqueeze(0), scale_factor=4, mode="nearest"
        ).squeeze(
            0
        )  # shape (n_channels, 4H, 4W)

        real_image = mat2img(upscaled_img_grid_real)  # [3, 4H, 4W]
        fake_image = mat2img(upscaled_img_grid_fake)  # [3, 4H, 4W]

        # add a black border on top, bottom and left of real image
        real_image = np.pad(
            real_image, ((0, 0), (2, 2), (2, 0)), "constant", constant_values=0
        )
        # add a gray border on right of real image
        real_image = np.pad(
            real_image, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=128
        )

        # add a black border on top, bottom and right of fake image
        fake_image = np.pad(
            fake_image, ((0, 0), (2, 2), (0, 2)), "constant", constant_values=0
        )
        # add a gray border on left of fake image
        fake_image = np.pad(
            fake_image, ((0, 0), (0, 0), (1, 0)), "constant", constant_values=196
        )

        real_fake_image = np.concatenate(
            [real_image, fake_image], axis=2
        )  # [3, 4H, 8W]

        images.append(real_fake_image)

    # we have 9 real images and 9 fake images
    # we want to make a 3x6 grid
    # we want to alternate real and fake images:
    # example first row: real1, fake1, real2, fake2, real3, fake3

    tot_H = images[0].shape[1] * 3
    tot_W = images[0].shape[2] * 3

    X = np.concatenate(images, axis=2).reshape(3, tot_H, tot_W)
    X = np.concatenate(
        [
            np.concatenate([images[0], images[1], images[2]], axis=2),
            np.concatenate([images[3], images[4], images[5]], axis=2),
            np.concatenate([images[6], images[7], images[8]], axis=2),
        ],
        axis=1,
    )

    # # map to RGB
    # third party
    from PIL import Image

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(X.transpose(1, 2, 0))

    pil_image.save("output_image.png")

    wandb.log(
        {"Real-vs-Fake": [wandb.Image(pil_image, caption="Real vs Fake")]},
        step=epoch,
        commit=False,
    )

    return


def save_examples2(real: torch.Tensor, fake: torch.Tensor, NVARS: int = 40) -> None:
    # everythinig is between -1 and 1
    L_CUT = NVARS + 1

    with torch.no_grad():

        # # map to RGB
        # third party
        import numpy as np
        from PIL import Image

        img_grid_real = torchvision.utils.make_grid(
            real[:9, :, :, :L_CUT], normalize=True, nrow=3
        )  # .unsqueeze(1) # shape (n_channels, H, W)
        img_grid_fake = torchvision.utils.make_grid(
            fake[:9, :, :, :L_CUT], normalize=True, nrow=3
        )  # .unsqueeze(1) # shape (n_channels, H, W)

        upscaled_img_grid_real = torch.nn.functional.interpolate(
            img_grid_real.unsqueeze(0), scale_factor=4, mode="nearest"
        ).squeeze(
            0
        )  # shape (n_channels, 4H, 4W)
        upscaled_img_grid_fake = torch.nn.functional.interpolate(
            img_grid_fake.unsqueeze(0), scale_factor=4, mode="nearest"
        ).squeeze(
            0
        )  # shape (n_channels, 4H, 4W)

        full_image = torch.cat(
            [upscaled_img_grid_real, upscaled_img_grid_fake], axis=-1
        )  # shape (n_channels, 4H, 8W)
        X = full_image.cpu().detach().numpy()

        # def grayscale_to_blue_red(x):
        #     if x < 0.5:
        #         return (int(255 * (1 - x)), int(255 * x), int(255 * x))
        #     else:
        #         return (int(255 * (1 - x)), int(255 * (1 - x)), int(255 * x))

        red_values = (255 * (1 - X[0])).astype(int)  # shape (H, W)
        blue_values = (255 * X[0]).astype(int)  # shape (H, W)
        green_values = (255 * X[0]).astype(int)  # shape (H, W)
        green_values[X[0] >= 0.5] = (255 * (1 - X[0, X[0] >= 0.5])).astype(int)
        # Apply the color mapping to the grayscale image

        # colored_image = np.apply_along_axis(grayscale_to_blue_red, 0, X[[0],:,:])
        colored_image = np.stack(
            [red_values, green_values, blue_values], axis=0
        )  # shape (3, H, W)

        colored_image = np.uint8(colored_image)
        colored_image[:, X[1, :, :] < 0.5] = 255

        # Convert the NumPy array to a PIL Image
        pil_image = Image.fromarray(colored_image.transpose(1, 2, 0))

        # pil_image.save("output_image.png")

        wandb.log({"full_image": [wandb.Image(pil_image, caption="real vs fake")]})

    return


# def mia_metrics_fn(disc, train_loader, val_loader):
#     # third party
#     from sklearn.metrics import (
#         accuracy_score,
#         average_precision_score,
#         f1_score,
#         roc_auc_score,
#     )

#     device = "cuda"

#     metrics = dict()

#     bce = torch.nn.BCEWithLogitsLoss(reduction="none")

#     disc_score_train = []
#     disc_score_val = []
#     disc_score_gen = []

#     # scores for train data
#     with torch.no_grad():
#         for idx, batch in enumerate(train_loader):
#             x = batch[0].to(device)  # (batch_size, 3, 256, 256)
#             y = batch[1].to(device)  # (batch_size, 3, 256, 256)
#             sta = batch[2].to(device)

#             # cur_batch_size = x.shape[0]

#             D_real = disc(x, y, sta=sta)
#             D_real_loss_train = torch.mean(
#                 bce(D_real, torch.ones_like(D_real)), dim=(1, 2, 3)
#             )
#             # y.shape, D_real.shape, D_real_loss_train.shape
#             disc_score_train.append(D_real_loss_train.cpu().numpy())

#     # scores for val data
#     with torch.no_grad():
#         for idx, batch in enumerate(val_loader):
#             x = batch[0].to(device)  # (batch_size, 3, 256, 256)
#             y = batch[1].to(device)  # (batch_size, 3, 256, 256)
#             sta = batch[2].to(device)

#             # cur_batch_size = x.shape[0]

#             D_real = disc(x, y, sta=sta)
#             D_real_loss_val = torch.mean(
#                 bce(D_real, torch.ones_like(D_real)), dim=(1, 2, 3)
#             )
#             # y.shape, D_real.shape, D_real_loss_train.shape
#             disc_score_val.append(D_real_loss_val.cpu().numpy())

#     disc_score_train = np.concatenate(disc_score_train)
#     disc_score_val = np.concatenate(disc_score_val)
#     disc_score_gen = np.concatenate(disc_score_gen)

#     y_score = np.concatenate(
#         [
#             disc_score_train,
#             disc_score_gen,
#             disc_score_val,
#         ]
#     )
#     y_true = np.concatenate(
#         [
#             np.ones_like(disc_score_train),
#             np.zeros_like(disc_score_gen),
#             np.zeros_like(disc_score_val),
#         ]
#     )

#     y_score.shape, y_true.shape

#     y_score.min(), y_score.max()

#     # # min-max normalize the scores
#     y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
#     y_score.min(), y_score.max()
#     # th = np.quantile(y_score, 1 - (disc_score_train.shape[0]) / y_score.shape[0])

#     # accuracy_score(y_true, y_score>0.5)
#     # roc_auc_score(y_true, y_score)
#     # f1_score(y_true, y_score>0.5)
#     # average_precision_score(y_true, y_score)

#     # put metrics in dict
#     metrics["MIA/accuracy"] = accuracy_score(y_true, y_score > 0.5)
#     metrics["MIA/roc_auc"] = roc_auc_score(y_true, y_score)
#     metrics["MIA/f1"] = f1_score(y_true, y_score > 0.5)
#     metrics["MIA/average_precision"] = average_precision_score(y_true, y_score)

#     # compute the baselines for accuracy, f1, average_precision
#     bl_accuracy = np.mean(y_true)
#     bl_f1 = 2 * bl_accuracy / (bl_accuracy + 1)
#     bl_average_precision = bl_accuracy

#     metrics["MIA/accuracy_baseline"] = bl_accuracy
#     metrics["MIA/f1_baseline"] = bl_f1
#     metrics["MIA/average_precision_baseline"] = bl_average_precision

#     return metrics


def prepro(
    df: pd.DataFrame, df_mean: pd.DataFrame, df_std: pd.DataFrame, state_vars: List
) -> pd.DataFrame:

    df = df.copy()
    # # missingness rate old
    # df_missing = df[['RecordID']+state_vars].groupby('RecordID').apply(lambda x:x.isnull().sum()/x.shape[0])[state_vars].reset_index()

    # missingness rate (normalized by max time)
    df_missing = (
        df[["id"] + ["timepoint"] + state_vars]
        .groupby("id")
        .apply(lambda x: x[state_vars].isnull().sum() / max(x.timepoint))[state_vars]
        .reset_index()
    )

    # change column names to "mr+{var}"
    df_missing.columns = ["id"] + ["mr_" + var for var in state_vars]

    # mean imputation for missing values
    df[state_vars] = df[state_vars].fillna(df[state_vars].mean())

    # standardize
    df[state_vars] = (df[state_vars] - df_mean) / df_std

    # group by id and compute statistics (min.max,mean,std) for each variable

    df2 = df.groupby("id")[state_vars].agg(["min", "max", "mean", "std"]).reset_index()

    # combine the names first and second level column indices

    df2.columns = ["_".join(col).strip() for col in df2.columns.values]

    df2.rename(columns={"id_": "id"}, inplace=True)

    df2 = df2.merge(df[["id", "outcome"]].drop_duplicates(), on=["id"], how="inner")

    df2 = df2.merge(df_missing, on=["id"], how="inner")

    return df2


def mat2df(
    all_dyn: np.ndarray,
    all_sta: np.ndarray,
    dynamic_processor: Any,
    static_processor: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    all_dyn = all_dyn * 0.5 + 0.5

    # print(dataset_schema.num_samples)
    MIDPOINT = 0.5
    # df_ts
    state_vars = dynamic_processor["features"]
    N_VARS = len(state_vars)
    N_VARS

    mat = all_dyn[:, 0, :, :N_VARS].reshape(-1, N_VARS)
    mask = all_dyn[:, 1, :, :N_VARS].reshape(-1, N_VARS) >= MIDPOINT

    df_ts = pd.DataFrame(mat, columns=state_vars)

    df_ts[state_vars] = (
        df_ts[state_vars] * (dynamic_processor["max"] - dynamic_processor["min"])
        + dynamic_processor["min"]
    )

    df_ts[state_vars] = (
        df_ts[state_vars] * dynamic_processor["std"] + dynamic_processor["mean"]
    )

    # setting nan values

    df_ts[~mask] = np.nan
    df_ts.isnull().sum().sum(), df_ts.shape
    df_ts.isnull().sum() / df_ts.shape[0] * 100

    # adding time
    time_array = np.arange(all_dyn.shape[2])
    # print(df_ts.shape, all_dyn.shape)
    col_time = np.tile(time_array, all_dyn.shape[0])

    # adding hospital
    # df_ts['Hospital'] = 0

    # adding RecordID
    col_recordid = np.repeat(np.arange(all_dyn.shape[0]), len(time_array))

    df_ts = pd.concat(
        [df_ts, pd.DataFrame({"timepoint": col_time, "id": col_recordid})], axis=1
    )
    # drop nan rows

    # OLD
    q_nan_rows_old = df_ts[state_vars].isnull().sum(axis=1) == N_VARS
    # # NEW
    # df_ts['INDIC'] = max_len.flatten()
    # q_nan_rows = df_ts['INDIC']==0
    # df_ts.drop('INDIC', axis=1, inplace=True)

    # print(f"dropping {q_nan_rows.sum()} rows")
    # print(f"old nans {q_nan_rows_old.sum()} rows")
    df_ts = df_ts[~q_nan_rows_old]
    df_ts.head()
    df_ts.shape

    df_ts = df_ts.dropna(subset=state_vars, how="all")

    # df_static

    features_encoded = static_processor["features_encoded"]
    static_types = static_processor["static_types"]
    static_features = static_processor["static_features"]

    df_static = pd.DataFrame()
    for i, (var, var_type) in enumerate(zip(static_features, static_types)):
        if var_type == "continuous":
            df_static[var] = all_sta[:, i]

            # de-standardize
            df_static[var] = (
                df_static[var] * static_processor[var]["std"]
                + static_processor[var]["mean"]
            )

        elif var_type == "binary":
            df_static[var] = (all_sta[:, i] > 0.5).astype(int)
        elif var_type == "categorical":
            i_categories = ["ICU" in x for x in features_encoded]
            df_static[var] = np.argmax(all_sta[:, i_categories], axis=1) + 1

    # add outcome
    df_static["outcome"] = all_sta[:, -1]

    # if len(features_encoded) == all_sta.shape[1]:

    #     df_demo = pd.DataFrame(all_sta, columns=features_encoded)
    # else:
    #     df_demo = pd.DataFrame(all_sta)
    # for col in features_encoded:
    #     if col in static_processor.keys():
    #         # de-standardize
    #         df_demo[col] = (
    #             df_demo[col] * static_processor[col]["std"]
    #             + static_processor[col]["mean"]
    #         )
    #         # print(col)
    # df_demo.columns
    # if "ICUType0" in df_demo.columns:
    #     df_demo["ICUType"] = (
    #         np.argmax(
    #             df_demo[["ICUType0", "ICUType1", "ICUType2", "ICUType3"]].values, axis=1
    #         )
    #         + 1
    #     )
    #     df_demo.drop(
    #         ["ICUType0", "ICUType1", "ICUType2", "ICUType3"], axis=1, inplace=True
    #     )

    # add RecordID
    df_static["id"] = np.arange(len(df_static))
    df_static.head()

    # make label column integer
    if "outcome" in df_static.columns:
        df_static["outcome"] = df_static["outcome"].astype(int)

    df_ts = df_ts.merge(df_static[["id", "outcome"]], on=["id"], how="inner")

    return df_ts, df_static


def find_last_epoch(path: str) -> int:

    all_files = os.listdir(path)

    # extract the last epoch
    last_epoch = max(
        # [int(file.split(".")[-2].split(".")[-1]) for file in all_files if "gen" in file]
        [int(file.split("_")[-1].split(".")[0]) for file in all_files if "gen" in file]
    )

    last_epoch

    return last_epoch


def create_df_demo(df: pd.DataFrame, demo_vars: List[str]) -> Tuple:
    # third party
    from sklearn.preprocessing import OneHotEncoder

    demo_vars

    df_demo = df[["RecordID", "Hospital"] + demo_vars].dropna(subset="Age")
    df_demo

    # replace -1 with NaN and the mean of each column
    df_demo.describe()
    df_demo[df_demo == -1] = np.nan
    df_demo.isnull().sum()
    df_demo.describe()
    for column in df_demo.columns:
        df_demo[column] = df_demo[column].fillna(df_demo[column].mean())
        # print(column)
    df_demo.isnull().sum()
    df_demo.describe()

    # standardize continuous variables
    demo_statistics = {}
    for column in demo_vars:
        if column in ["Age", "Height", "Weight", "HospAdmTime"]:
            demo_statistics[column] = {
                "mean": df_demo[column].mean(),
                "std": df_demo[column].std(),
            }

            df_demo[column] = (df_demo[column] - df_demo[column].mean()) / df_demo[
                column
            ].std()

    # discritze ICUType
    if "ICUType" in demo_vars:

        ohe = OneHotEncoder()
        transformed = ohe.fit_transform(df_demo[["ICUType"]])
        transformed
        mat_ICU_enc = pd.DataFrame.sparse.from_spmatrix(transformed).values
        new_cols = ["ICUType" + str(i) for i in range(mat_ICU_enc.shape[1])]
        df_demo[new_cols] = mat_ICU_enc

        df_demo = df_demo.drop(columns="ICUType")

    df_demo.rename(columns={"RecordID": "id"}, inplace=True)

    df_demo

    demo_vars_new = list(df_demo.columns)
    demo_vars_new.remove("id")
    demo_vars_new.remove("Hospital")
    demo_vars_new

    df_demo["dict_demo"] = df_demo[demo_vars_new].apply(lambda x: list(x), axis=1)

    df_demo.iloc[0]["dict_demo"]

    dict_map_demos = {k: i for i, k in enumerate(demo_vars_new)}
    dict_map_demos

    return df_demo, dict_map_demos, demo_statistics


# def ffill(a,mask,limit=6):
#     mask = mask * 0.5+0.5
#     first = a[:,0,:]
#     for i in range(limit):
#         a = torch.roll(a,1,dims=1)*(1-mask)+mask*a
#         mask = (a>0).float()
#     a[:,0,:] = first
#     return a


def ffill(a: torch.Tensor, mask: torch.Tensor, limit: int = 6) -> torch.Tensor:
    mask = (mask > 0).float()
    first = a[:, 0, :]
    for i in range(limit):
        a = torch.roll(a, 1, dims=1) * (1 - mask) + mask * a
        mask = (a > 0).float()
    a[:, 0, :] = first
    return a


def handle_cgan(
    df_demo: pd.DataFrame,
    df_filt: pd.DataFrame,
    state_vars: List,
    demo_vars_new: List,
    granularity: int = 1,
    target_dim: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    var_old = [
        "Albumin",
        "ALP",
        "ALT",
        "AST",
        "Bilirubin",
        "BUN",
        "Cholesterol",
        "Creatinine",
        "DiasABP",
        "FiO2",
        "GCS",
        "Glucose",
        "HCO3",
        "HCT",
        "HR",
        "K",
        "Lactate",
        "Mg",
        "MAP",
        "MechVent",
        "Na",
        "NIDiasABP",
        "NIMAP",
        "NISysABP",
        "PaCO2",
        "PaO2",
        "pH",
        "Platelets",
        "RespRate",
        "SaO2",
        "SysABP",
        "Temp",
        "TroponinI",
        "TroponinT",
        "Urine",
        "WBC",
    ]

    print("set(state_vars)-set(var_old) => ", set(state_vars) - set(var_old))
    print("set(var_old)-set(state_vars) => ", set(var_old) - set(state_vars))

    # df_demo, demo_dict = create_df_demo(df_demo, demo_vars)
    # print(demo_dict)
    # print(df_demo.columns)
    print(demo_vars_new)
    sta = df_demo[demo_vars_new].rename(columns={"In-hospital_death": "Label"})
    # sta['Label'] = sta['Label'].astype(float)
    # sta["Gender"] = sta["Gender"].astype(int)
    # sta["Label"] = sta["Label"].astype(int)

    # print(sta)
    all_sta = torch.from_numpy(sta.values)
    print(all_sta.shape)

    dyn = []

    grouped = df_filt.groupby("RecordID")

    # Create a list of DataFrames
    dyn = [
        group[["Time"] + state_vars].rename(columns={"Time": "time"})
        for _, group in grouped
    ]

    all_times = []
    all_dyn = []
    all_masks = []

    for SAMPLE in dyn:
        # time = SAMPLE.time.values
        # times_padded = pd.DataFrame({'time':np.arange(0,   int(max(time)),granularity)})

        # print(time,len(time),max(time))
        # print(times_padded.values.flatten(),len(times_padded.values.flatten()))

        # create a np array from 0 to max(time) with granularity 0.5 ()including max(time)) use linspace
        time_padded = np.linspace(
            0, int(target_dim * granularity) - granularity, target_dim
        )  # shape is (target_dim,)

        df_time_padded = pd.DataFrame({"time": time_padded})
        temp = df_time_padded.merge(
            SAMPLE, how="outer", on="time"
        )  # .sort_values(by='time_padded').

        dyn_padded = temp[state_vars].fillna(0).values
        mask_padded = temp[state_vars].notnull().astype(int).values

        all_times.append(torch.from_numpy(time_padded)[:target_dim])
        all_dyn.append(torch.from_numpy(dyn_padded)[:target_dim])
        all_masks.append(torch.from_numpy(mask_padded)[:target_dim])
        # print(time_padded.shape,dyn_padded.shape,mask_padded.shape)

    # return all_masks,all_dyn,all_times

    all_masks = torch.stack(all_masks, dim=0)
    all_dyn = torch.stack(all_dyn, dim=0)
    all_times = torch.stack(all_times, dim=0)

    # PADDING
    # target_dim = TARGET_DIM
    padding_needed = target_dim - len(state_vars)

    all_masks_padded = (
        torch.nn.functional.pad(all_masks, (0, padding_needed)).unsqueeze(1).float()
    )  # add channel dim
    all_dyn_padded = (
        torch.nn.functional.pad(all_dyn, (0, padding_needed)).unsqueeze(1).float()
    )  # add channel dim
    # all_times_padded = torch.nn.functional.pad(all_times, (0, padding_needed))
    all_masks_padded.shape
    all_dyn_padded.shape

    all_data = torch.stack([all_dyn_padded, all_masks_padded], dim=1)
    all_data.shape

    # SCALE all_data to [-1,1]
    all_masks_padded = all_masks_padded * 2 - 1
    all_dyn_padded = all_dyn_padded * 2 - 1
    all_dyn_padded[all_masks_padded < 0] = 0
    # all_data = all_data*2-1
    # all_data[:,0,:,:][all_data[:,1,:,:]<0]=0

    return all_masks_padded.float(), all_dyn_padded.float(), all_sta.float()


def custom_process(
    df_demo: pd.DataFrame,
    df_filt: pd.DataFrame,
    train_ids: Any,
    state_vars: List,
    gan_demovars: List,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:

    df_demo_train = df_demo[df_demo["RecordID"].isin(train_ids)].copy()
    df_filt_train = df_filt[df_filt["RecordID"].isin(train_ids)].copy()

    df_demo_test = df_demo[~df_demo["RecordID"].isin(train_ids)].copy()
    df_filt_test = df_filt[~df_filt["RecordID"].isin(train_ids)].copy()

    print(
        df_demo_train.shape, df_demo_test.shape, df_filt_train.shape, df_filt_test.shape
    )
    # Normalize state variables

    # 1] normalized cols in state_vars from df_filt
    state_preprocess = {
        "mean": df_filt_train[state_vars].mean(),
        "std": df_filt_train[state_vars].std(),
    }

    df_filt_train[state_vars] = (
        df_filt_train[state_vars] - state_preprocess["mean"]
    ) / state_preprocess["std"]
    df_filt_test[state_vars] = (
        df_filt_test[state_vars] - state_preprocess["mean"]
    ) / state_preprocess["std"]

    # 2] set outliers to nan
    df_filt_train[state_vars] = df_filt_train[state_vars].apply(
        lambda x: x.mask(x.sub(x.mean()).div(x.std()).abs().gt(3))
    )
    df_filt_test[state_vars] = df_filt_test[state_vars].apply(
        lambda x: x.mask(x.sub(x.mean()).div(x.std()).abs().gt(3))
    )

    # 3] now do min-max normalization # between 0 and 1
    state_preprocess["min"] = df_filt_train[state_vars].min()
    state_preprocess["max"] = df_filt_train[state_vars].max()

    df_filt_train[state_vars] = (
        df_filt_train[state_vars] - state_preprocess["min"]
    ) / (state_preprocess["max"] - state_preprocess["min"])
    df_filt_test[state_vars] = (df_filt_test[state_vars] - state_preprocess["min"]) / (
        state_preprocess["max"] - state_preprocess["min"]
    )

    # # 4] scale to [-1,1]
    # df_filt_train[state_vars] = df_filt_train[state_vars]*2-1
    # df_filt_test[state_vars] = df_filt_test[state_vars]*2-1

    # Normalize demo variables

    df_demo_train, demo_dict, demo_preprocess = create_df_demo(
        df_demo_train, gan_demovars
    )
    demo_preprocess["demo_vars_new"] = list(demo_dict.keys())
    if len(df_demo_test) > 0:
        df_demo_test, _, _ = create_df_demo(df_demo_test, gan_demovars)

    # print(list(demo_preprocess['demo_vars_new']))
    return (
        df_demo_train,
        df_demo_test,
        df_filt_train,
        df_filt_test,
        state_preprocess,
        demo_preprocess,
    )


def dl_runs(all_runs: List, selected_tag: Any = None) -> pd.DataFrame:

    if selected_tag is None:
        selected_tag = []
    if type(selected_tag) == str:
        selected_tag = [selected_tag]

    # selected_tag = list(selected_tag) if selected_tag is not None else [] #

    summary_list, config_list, name_list, path_list = [], [], [], []
    for run in all_runs:

        # if len(set(selected_tag).intersection(set(run.tags)))==0:
        #     continue
        # print(run.tags)
        # if (selected_tag != run.tags) and (selected_tag is not None):
        #     continue
        # if (len(set(selected_tag)-set(run.tags))>0) and (selected_tag is not None):
        #     continue

        # run.tags contains all of selected_tag
        if (len(set(selected_tag) - set(run.tags)) > 0) and (selected_tag is not None):
            continue

        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.

        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        path_list.append(run.path)

    runs_df = pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
            "path": path_list,
        }
    )

    # runs_df.to_csv("project.csv")
    return runs_df


def fun_cohensd(
    df_pred: pd.DataFrame, df_true: pd.DataFrame, state_vars: List
) -> pd.Series:
    df_stat = pd.Series(index=state_vars, dtype=str)

    # categorize cohens d into small, medium, large
    def cohens_d_cat(x: float) -> str:
        if x < 0.2:
            return "small"
        elif x < 0.5:
            return "medium"
        else:
            return "large"

    for col in state_vars:
        col1 = df_true[col].dropna()
        col2 = df_pred[col].dropna()

        # compute cohens d
        d = (col1.mean() - col2.mean()) / (
            np.sqrt((col1.std() ** 2 + col2.std() ** 2) / 2)
        )
        df_stat.loc[col] = np.abs(d)
        if cohens_d_cat(np.abs(d)) == "medium":
            df_stat.loc[col] = "*"
        elif cohens_d_cat(np.abs(d)) == "large":
            df_stat.loc[col] = "**"
        else:
            df_stat.loc[col] = ""
    return df_stat


def plot_corr(
    df_train_real: pd.DataFrame,
    df_train_fake: pd.DataFrame,
    df_test: pd.DataFrame,
    state_vars: List,
    corr_method: str = "",
    corr_th: float = 0.2,
) -> Any:
    def impute(df: pd.DataFrame) -> pd.DataFrame:

        # if CORR_METHOD=='ffill':
        df = df.copy()
        df[state_vars] = (
            df[["RecordID"] + state_vars]
            .groupby("RecordID")
            .fillna(method="ffill", limit=6)
        )

        # # mean imputation
        # for col in state_vars:
        #     df[col].fillna(df[col].mean(), inplace=True)

        return df

    def corr_agg(df: pd.DataFrame) -> np.ndarray:

        df = impute(df)
        temp = df.groupby("RecordID")[state_vars].corr()
        grouped = [group.droplevel(0).values for _, group in temp.groupby("RecordID")]

        corr = np.stack(grouped, axis=0)
        corr.shape

        # corr[np.abs(corr)<0.2]=0
        # set np.nan to zero
        corr[np.isnan(corr)] = 0

        # set upper triangle and diagonal to zero
        for i in range(corr.shape[0]):
            corr[i][np.triu_indices_from(corr[i], k=0)] = 0

        corr = corr.mean(0)

        return corr

    def compute_temp_corr(
        mat_true: np.ndarray, mat_syn: np.ndarray, th: float = 0.2
    ) -> float:
        norm_const = mat_true.shape[0] * (mat_true.shape[0] - 1) / 2
        mat_true = mat_true.copy()
        mat_syn = mat_syn.copy()

        mat_true[np.abs(mat_true) < th] = 0
        mat_syn[np.abs(mat_syn) < th] = 0

        # set upper triangle and diagonal to zero
        mat_true[np.triu_indices_from(mat_true, k=0)] = 0
        mat_syn[np.triu_indices_from(mat_syn, k=0)] = 0

        # set nans to zero
        mat_true[np.isnan(mat_true)] = 0
        mat_syn[np.isnan(mat_syn)] = 0

        x = np.mean((mat_true - mat_syn) ** 2)

        # compute L1 loss
        x = np.sum(np.abs(mat_true - mat_syn)) / norm_const

        # # compute frobenius norm
        # x = np.linalg.norm(mat_true-mat_syn)

        return x

    if corr_method == "ffill":

        corr_train = impute(df_train_real)[state_vars].corr().values
        corr_val = impute(df_test)[state_vars].corr().values
        corr_gen = impute(df_train_fake)[state_vars].corr().values
    elif corr_method == "agg":
        corr_train = corr_agg(df_train_real)
        corr_val = corr_agg(df_test)
        corr_gen = corr_agg(df_train_fake)
    else:
        corr_train = df_train_real[state_vars].corr().values
        corr_val = df_test[state_vars].corr().values
        corr_gen = df_train_fake[state_vars].corr().values

    # print MSE of correlation matrices

    print(
        f"MSE of correlation matrices: {compute_temp_corr(corr_train[:,:], corr_gen[:,:],th=0):.3f}"
    )
    print(
        f"MSE of correlation matrices BL: {compute_temp_corr(corr_train[:,:], corr_val[:,:],th=0):.3f}"
    )

    mask = np.logical_or(np.abs(corr_train) < corr_th, np.abs(corr_val) < corr_th)
    mask = np.abs(corr_train) < corr_th
    # corr_train[np.abs(corr_train)<corr_th]=0
    # corr_val[np.abs(corr_val)<corr_th]=0
    # corr_gen[np.abs(corr_gen)<corr_th]=0

    # set upper triangle and diagonal to zero
    corr_train[np.triu_indices_from(corr_train, k=0)] = 0
    corr_val[np.triu_indices_from(corr_val, k=0)] = 0
    corr_gen[np.triu_indices_from(corr_gen, k=0)] = 0
    y_names = state_vars.copy()
    y_names.reverse()

    mask_nan_gen = np.isnan(corr_gen)

    corr_diff = np.abs(corr_gen - corr_train)  # /(corr_train+1e-9)
    # set nans to zero
    corr_diff[mask] = 0
    # corr_diff[np.isnan(corr_diff)]=-1
    corr_diff[mask_nan_gen] = -1
    corr_diff[np.abs(corr_diff) < corr_th] = 0

    # mirror horizontally (if using plotly go)
    corr_train = corr_train[::-1]
    corr_val = corr_val[::-1]
    corr_gen = corr_gen[::-1]
    corr_diff = corr_diff[::-1]

    # plot all in a subplot 2 by 2
    # each subplot should be square
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Train", "Val", "Gen", "Diff"))

    sub_train = go.Heatmap(
        z=corr_train, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_train, row=1, col=1)

    sub_val = go.Heatmap(
        z=corr_val, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_val, row=1, col=2)

    sub_gen = go.Heatmap(
        z=corr_gen, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_gen, row=2, col=1)

    sub_diff = go.Heatmap(
        z=corr_diff, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_diff, row=2, col=2)

    # set hiehgts and width of each subplot to be equal
    _ = fig.update_layout(height=1200, width=1100, title_text="Correlation matrices")

    #     _ = fig.update_layout(
    #     # subplot_titles=['Heatmap 1', 'Heatmap 2', 'Heatmap 3', 'Heatmap 4'],
    #     # grid=dict(rows=2, columns=2),  # 2x2 grid for 4 subplots
    #     row_heights=[1, 1],  # Set the relative heights (1:1 ratio for both rows)
    #     column_widths=[1, 1]  # Set the relative widths (1:1 ratio for both columns)
    # )

    for i in range(1, 5):
        _ = fig.update_yaxes(scaleanchor=f"x{str(i)}", scaleratio=1)
        _ = fig.update_xaxes(scaleanchor=f"y{str(i)}", scaleratio=1)

    return fig, (sub_train, sub_val, sub_gen, sub_diff)


def compute_temp_corr(
    df_train_real: pd.DataFrame,
    df_train_fake: pd.DataFrame,
    df_test: pd.DataFrame,
    state_vars: List,
    corr_method: str = "",
    corr_th: float = 0.2,
) -> dict:
    def impute(df: pd.DataFrame) -> pd.DataFrame:

        # if CORR_METHOD=='ffill':
        df = df.copy()
        df[state_vars] = (
            df[["id"] + state_vars].groupby("id").fillna(method="ffill", limit=6)
        )

        # # mean imputation
        # for col in state_vars:
        #     df[col].fillna(df[col].mean(), inplace=True)

        return df

    def corr_agg(df: pd.DataFrame) -> np.ndarray:

        df = impute(df)
        temp = df.groupby("id")[state_vars].corr()
        grouped = [group.droplevel(0).values for _, group in temp.groupby("id")]

        corr = np.stack(grouped, axis=0)
        corr.shape

        # corr[np.abs(corr)<0.2]=0
        # set np.nan to zero
        corr[np.isnan(corr)] = 0

        # set upper triangle and diagonal to zero
        for i in range(corr.shape[0]):
            corr[i][np.triu_indices_from(corr[i], k=0)] = 0

        corr = corr.mean(0)

        return corr

    def compute_metric(
        mat_true: np.ndarray, mat_syn: np.ndarray, th: float = 0.2
    ) -> float:
        norm_const = mat_true.shape[0] * (mat_true.shape[0] - 1) / 2

        mat_true2 = mat_true.copy()
        mat_syn2 = mat_syn.copy()

        mat_true2[np.abs(mat_true2) < th] = 0
        mat_syn2[np.abs(mat_syn2) < th] = 0

        # set upper triangle and diagonal to zero
        mat_true2[np.triu_indices_from(mat_true2, k=0)] = 0
        mat_syn2[np.triu_indices_from(mat_syn2, k=0)] = 0

        # set nans to zero
        mat_true2[np.isnan(mat_true2)] = 0
        mat_syn2[np.isnan(mat_syn2)] = 0

        x = np.mean((mat_true2 - mat_syn2) ** 2)

        # compute frobenius norm
        x = np.linalg.norm(mat_true2 - mat_syn2)

        # compute L1 loss
        x = np.sum(np.abs(mat_true2 - mat_syn2)) / norm_const

        return x

    if corr_method == "ffill":

        corr_train = impute(df_train_real)[state_vars].corr().values
        corr_val = impute(df_test)[state_vars].corr().values
        corr_gen = impute(df_train_fake)[state_vars].corr().values
    elif corr_method == "agg":
        corr_train = corr_agg(df_train_real)
        corr_val = corr_agg(df_test)
        corr_gen = corr_agg(df_train_fake)
    else:
        corr_train = df_train_real[state_vars].corr().values
        corr_val = df_test[state_vars].corr().values
        corr_gen = df_train_fake[state_vars].corr().values

    # print MSE of correlation matrices

    # print(f"MSE of correlation matrices: {compute_metric(corr_train, corr_gen):.3f}")
    # print(f"MSE of correlation matrices BL: {compute_metric(corr_train, corr_val):.3f}")

    metric = {
        "TCD[Train-Synthetic]/stats.tc_corr": compute_metric(corr_train, corr_gen),
        "TCD[Train-Test]/stats.tc_corr": compute_metric(corr_train, corr_val),
    }

    # log to wandb
    # mirror horizontally (if using plotly go)
    mat_true = corr_train.copy()
    mat_syn = corr_gen.copy()
    th = corr_th

    mat_true[np.abs(mat_true) < th] = 0
    mat_syn[np.abs(mat_syn) < th] = 0

    # set upper triangle and diagonal to zero
    mat_true[np.triu_indices_from(mat_true, k=0)] = 0
    mat_syn[np.triu_indices_from(mat_syn, k=0)] = 0

    # set nans to zero
    mat_true[np.isnan(mat_true)] = 0
    mat_syn[np.isnan(mat_syn)] = 0

    # mirror horizontally (if using plotly go)
    mat_true = mat_true[::-1]
    mat_syn = mat_syn[::-1]

    y_names = state_vars.copy()
    y_names.reverse()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Real", "Synthetic"))

    sub_train = go.Heatmap(
        z=mat_true, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_train, row=1, col=1)

    sub_val = go.Heatmap(
        z=mat_syn, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_val, row=1, col=2)
    wandb.log({"Correlation Matrices": fig})

    return metric


def plot_tsne(REAL: torch.Tensor, FAKE: torch.Tensor, N: int = 10000) -> go.Figure:
    # for t-SNE
    # third party
    from MulticoreTSNE import MulticoreTSNE as TSNE

    X = np.concatenate([REAL[:N], FAKE[:N]], axis=0)  # [2*bs, hidden_dim]

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=10, n_jobs=4)

    X_tsne = tsne.fit_transform(X)  # [2*bs, 2]

    N2 = int(X.shape[0] / 2)

    fig_tsne = go.Figure()
    _ = fig_tsne.add_trace(
        go.Scatter(x=X_tsne[:N2, 0], y=X_tsne[:N2, 1], mode="markers", name="real")
    )
    _ = fig_tsne.add_trace(
        go.Scatter(x=X_tsne[N2:, 0], y=X_tsne[N2:, 1], mode="markers", name="fake")
    )

    # fig_tsne.show()

    # # plot histogram
    # REAL2 = REAL[:,3].flatten()
    # FAKE2 = FAKE[:,3].flatten()
    # fig_hist = go.Figure()
    # _ = fig_hist.add_trace(go.Histogram(x=REAL2, nbinsx=100, name='real'))
    # _ = fig_hist.add_trace(go.Histogram(x=FAKE2, nbinsx=100, name='fake'))
    # fig_hist.show()

    return fig_tsne


def xgboost_embeddings(
    df: pd.DataFrame, state_vars: List, df_base: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.Series]:

    if df_base is not None:
        df_mean, df_std = df_base[state_vars].mean(), df_base[state_vars].std()
    else:
        df_mean, df_std = df[state_vars].mean(), df[state_vars].std()

    print("[info] Preprocessing ")
    df_pro = prepro(df, df_mean, df_std, state_vars)

    features_names = df_pro.columns.tolist()
    features_names.remove("id")
    features_names.remove("outcome")

    X = df_pro[features_names]
    y = df_pro["outcome"]

    return X, y
