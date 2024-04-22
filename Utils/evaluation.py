# from utils import fun_cohensd, plot_corr


import os
import wandb
from dotenv import load_dotenv

import hydra
from omegaconf import DictConfig, OmegaConf

import pandas as pd

from Utils.utils import xgboost_embeddings, compute_temp_corr, plot_tsne, save_examples


from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
)

import lightgbm as lgb

# For MIA-kNN
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import gaussian_kde
from scipy.stats import entropy, wasserstein_distance

import numpy as np


import torch

UTILITY_MODE = ["TRTR", "TSTR"]


def compute_utility(
    X_train_real,
    X_train_fake,
    X_test_real,
    X_test_fake,
    y_train_real,
    y_train_fake,
    y_test_real,
    y_test_fake,
):

    metrics = dict()
    if "TRTR" in UTILITY_MODE:
        # TRTR

        # compute scale_pos_weight
        scale_pos_weight = (y_train_real == 0).sum() / (y_train_real == 1).sum()

        print("\n[info] TRTR")

        # fit real model
        model_real = lgb.LGBMClassifier(
            random_state=42, scale_pos_weight=scale_pos_weight, verbosity=1
        )
        model_real.fit(X_train_real, y_train_real)

        # eval model
        y_pred = model_real.predict(X_test_real)
        y_score = model_real.predict_proba(X_test_real)[:, 1]
        y_true = y_test_real.values

        print(f"AUROC: {roc_auc_score(y_true, y_score)}")
        print(f"AUPRC: {average_precision_score(y_true, y_score)}")
        print(f"F1: {f1_score(y_true, y_pred)}")

        metrics.update(
            {
                "TRTR/AUROC": roc_auc_score(y_true, y_score),
                "TRTR/AUPRC": average_precision_score(y_true, y_score),
                "TRTR/F1": f1_score(y_true, y_pred),
            }
        )

    if "TSTR" in UTILITY_MODE:

        # fake train

        # compute scale_pos_weight
        scale_pos_weight = (y_train_fake == 0).sum() / (y_train_fake == 1).sum()

        print("\n[info] TSTR")
        model_fake = lgb.LGBMClassifier(
            random_state=42, scale_pos_weight=scale_pos_weight, verbosity=1
        )
        model_fake.fit(X_train_fake, y_train_fake)

        # eval model
        y_pred = model_fake.predict(X_test_fake)
        y_score = model_fake.predict_proba(X_test_fake)[:, 1]
        y_true = y_test_fake.values

        print(f"AUROC: {roc_auc_score(y_true, y_score)}")
        print(f"AUPRC: {average_precision_score(y_true, y_score)}")
        print(f"F1: {f1_score(y_true, y_pred)}")

        metrics.update(
            {
                "TSTR/AUROC": roc_auc_score(y_true, y_score),
                "TSTR/AUPRC": average_precision_score(y_true, y_score),
                "TSTR/F1": f1_score(y_true, y_pred),
            }
        )

    return metrics


def compute_synthcity(X_real, X_fake, X_test):
    # import sys
    # sys.path.insert(0, "/mlodata1/hokarami/synthcity/src")
    from synthcity.plugins.core.dataloader import GenericDataLoader
    from synthcity.metrics import Metrics
    from synthcity.metrics.scores import ScoreEvaluator

    from pathlib import Path

    scores = ScoreEvaluator()
    X = GenericDataLoader(
        X_real,
        target_column="outcome",
    )
    X_test = GenericDataLoader(
        X_test,
        target_column="outcome",
    )
    X_syn = GenericDataLoader(
        X_fake,
        target_column="outcome",
    )
    X_ref_syn = X_syn
    X_augmented = None
    selected_metrics = {
        "privacy": [
            "delta-presence",
            "k-anonymization",
            "k-map",
            "distinct l-diversity",
            "identifiability_score",
            "DomiasMIA_BNAF",
            "DomiasMIA_KDE",
            "DomiasMIA_prior",
        ]
    }
    selected_metrics = {
        "stats": [
            "alpha_precision",
        ],
        "privacy": ["identifiability_score"],
    }
    # selected_metrics={
    #     'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
    #             'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
    #             'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
    #             'detection': ['detection_xgb', 'detection_mlp', 'detection_gmm', 'detection_linear'],
    #     'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']}

    selected_metrics = {
        "stats": [
            "alpha_precision",
            # 'jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy',
            "prdc",
            # 'survival_km_distance',
            # 'wasserstein_dist'
        ],
        #         'stats': [
        #             'alpha_precision',
        #             'jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'
        #         ],
        "privacy": [
            #  'delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity',
            "identifiability_score"
        ],
    }

    selected_metrics = {
        "stats": [
            # 'alpha_precision',
            "prdc",
        ],
        "privacy": ["identifiability_score"],
    }

    n_repeats = 1

    # # DEBUG
    # scores = ScoreEvaluator()
    # from sklearn.datasets import load_diabetes
    # X, y = load_diabetes(return_X_y=True, as_frame=True)
    # X["target"] = y

    # loader = GenericDataLoader(
    #     X,
    #     target_column="target",
    #     sensitive_columns=["sex"],
    # )

    # X = loader.train()
    # X_test = loader.test()
    # X_syn = loader
    # X_ref_syn = loader
    # X_augmented = None

    print("\n[info] computing synthcity metrics X_train and X_test")
    scores = ScoreEvaluator()

    for _ in range(n_repeats):
        evaluation = Metrics.evaluate(
            X,
            X_test,
            X,
            X_ref_syn,
            X_augmented,
            metrics=selected_metrics,
            task_type="classification",
            workspace=Path("workspace"),
            use_cache=False,
        )
        mean_score = evaluation["mean"].to_dict()
        errors = evaluation["errors"].to_dict()
        duration = evaluation["durations"].to_dict()
        direction = evaluation["direction"].to_dict()

        for key in mean_score:
            scores.add(
                key,
                mean_score[key],
                errors[key],
                duration[key],
                direction[key],
            )
    metrics_syn_test = scores.to_dataframe()["mean"].to_dict()

    metrics_syn_test = {
        ("Synthcity[Train-Test]/" + k): v for k, v in metrics_syn_test.items()
    }

    print("\n[info] computing synthcity metrics X_train and X_syn")
    scores = ScoreEvaluator()

    for _ in range(n_repeats):
        evaluation = Metrics.evaluate(
            X,
            X_syn,
            X,
            X_ref_syn,
            X_augmented,
            metrics=selected_metrics,
            task_type="classification",
            workspace=Path("workspace"),
            use_cache=False,
        )
        mean_score = evaluation["mean"].to_dict()
        errors = evaluation["errors"].to_dict()
        duration = evaluation["durations"].to_dict()
        direction = evaluation["direction"].to_dict()

        for key in mean_score:
            scores.add(
                key,
                mean_score[key],
                errors[key],
                duration[key],
                direction[key],
            )
    metrics_syn = scores.to_dataframe()["mean"].to_dict()

    metrics_syn = {
        ("Synthcity[Train-Synthetic]/" + k): v for k, v in metrics_syn.items()
    }

    # combine two metrics
    metrics = {**metrics_syn_test, **metrics_syn}

    # print(metrics)

    return metrics


def compute_nnaa(REAL, FAKE, TEST):

    def ff(A, B, self=False):
        # print(A.shape, B.shape)
        np.sum(A**2, axis=1).reshape(A.shape[0], 1).shape, np.sum(B.T**2, axis=0).shape
        a = np.sum(A**2, axis=1).reshape(A.shape[0], 1) + np.sum(
            B.T**2, axis=0
        ).reshape(1, B.shape[0])
        b = np.dot(A, B.T) * 2
        distance_matrix = a - b
        a.shape, b.shape, distance_matrix.shape
        np.min(distance_matrix, axis=0)
        if self == True:
            np.fill_diagonal(distance_matrix, np.inf)
        # print(distance_matrix[:5,:5])
        # print(np.min(distance_matrix[:5,:5], axis=1))
        min_dist_AB = np.min(distance_matrix, axis=1)
        min_dist_BA = np.min(distance_matrix, axis=0)

        return min_dist_AB, min_dist_BA

    distance_TT, _ = ff(REAL, REAL, self=True)
    distance_EE, _ = ff(TEST, TEST, self=True)
    distance_SS, _ = ff(FAKE, FAKE, self=True)

    distance_TS, distance_ST = ff(REAL, FAKE)
    distance_ES, distance_SE = ff(TEST, FAKE)
    distance_TE, distance_ET = ff(REAL, TEST)

    distance_TS.shape, distance_ST.shape, distance_TT.shape, distance_SS.shape
    distance_EE.shape, distance_SE.shape, distance_ES.shape

    aa_train = (
        np.sum(distance_TS > distance_TT) / distance_TT.shape[0]
        + np.sum(distance_ST > distance_SS) / distance_SS.shape[0]
    ) / 2

    aa_test = (
        np.sum(distance_ES > distance_EE) / distance_EE.shape[0]
        + np.sum(distance_SE > distance_SS) / distance_SS.shape[0]
    ) / 2

    aa_train_test = (
        np.sum(distance_TE > distance_TT) / distance_TT.shape[0]
        + np.sum(distance_ET > distance_EE) / distance_EE.shape[0]
    ) / 2

    metrics_sym = {
        "Adv ACC/AA_train_syn": aa_train,
        "Adv ACC/AA_test_syn": aa_test,
        "Adv ACC/AA_train_test": aa_train_test,
        "Adv ACC/NNAA": (aa_train - aa_test),
    }

    metrics_asym = {
        "Train-Fake": np.sum(distance_TS > distance_TT) / distance_TT.shape[0]
        - distance_TT.shape[0] / (distance_TT.shape[0] + distance_SS.shape[0]),
        "Fake-Train": np.sum(distance_ST > distance_SS) / distance_SS.shape[0]
        - distance_SS.shape[0] / (distance_TT.shape[0] + distance_SS.shape[0]),
        "Test-Fake": np.sum(distance_ES > distance_EE) / distance_EE.shape[0]
        - distance_EE.shape[0] / (distance_EE.shape[0] + distance_SS.shape[0]),
        "Fake-Test": np.sum(distance_SE > distance_SS) / distance_SS.shape[0]
        - distance_SS.shape[0] / (distance_EE.shape[0] + distance_SS.shape[0]),
        "Train-Test": np.sum(distance_TE > distance_TT) / distance_TT.shape[0]
        - distance_TT.shape[0] / (distance_TT.shape[0] + distance_EE.shape[0]),
        "Test-Train": np.sum(distance_ET > distance_EE) / distance_EE.shape[0]
        - distance_EE.shape[0] / (distance_TT.shape[0] + distance_EE.shape[0]),
    }

    metrics_asym_bl = {
        # "Train-Fake-bl": distance_TT.shape[0]/(distance_TT.shape[0]+distance_SS.shape[0]),
        # "Fake-Train-bl": distance_SS.shape[0]/(distance_TT.shape[0]+distance_SS.shape[0]),
        # "Test-Fake-bl": distance_EE.shape[0]/(distance_EE.shape[0]+distance_SS.shape[0]),
        # "Fake-Test-bl": distance_SS.shape[0]/(distance_EE.shape[0]+distance_SS.shape[0]),
        # "Train-Test-bl": distance_TT.shape[0]/(distance_TT.shape[0]+distance_EE.shape[0]),
        # "Test-Train-bl": distance_EE.shape[0]/(distance_TT.shape[0]+distance_EE.shape[0]),
    }
    return metrics_sym, metrics_asym, metrics_asym_bl


def compute_mia_knn(REAL, FAKE, TEST):
    knn = KNeighborsClassifier(n_neighbors=1)

    X = np.concatenate(
        [
            # REAL,
            FAKE
        ],
        axis=0,
    )  # [2*bs, hidden_dim]
    y = np.concatenate(
        [
            # np.ones(REAL.shape[0]),
            np.zeros(FAKE.shape[0])
        ],
        axis=0,
    )  # [2*bs, hidden_dim]

    knn.fit(X, y)

    test_nearest_dist, test_nearest_ids = knn.kneighbors(TEST, return_distance=True)
    train_nearest_dist, train_nearest_ids = knn.kneighbors(REAL, return_distance=True)

    if test_nearest_dist.shape[1] > 1:  # if more than 1 neighbor
        test_nearest_dist = test_nearest_dist.mean(1)
        train_nearest_dist = train_nearest_dist.mean(1)

    # fit non-parametric density
    kde_train = gaussian_kde(train_nearest_dist.flatten())
    kde_test = gaussian_kde(test_nearest_dist.flatten())

    def jensen_shannon_divergence(p, q):
        # Calculate the average distribution
        m = 0.5 * (p + q)

        # Calculate the Jensen-Shannon Divergence
        jsd = 0.5 * (entropy(p, m) + entropy(q, m))
        return jsd

    max_dist = max(train_nearest_dist.max(), test_nearest_dist.max())
    x_values = np.linspace(0, max_dist, 100)
    pdf_train = kde_train(x_values)
    pdf_test = kde_test(x_values)

    wasserstein_distance(pdf_train, pdf_test)
    jensen_shannon_divergence(pdf_train, pdf_test)

    metrics = {
        "MIA/WD": wasserstein_distance(pdf_train, pdf_test),
        "MIA/JSD": jensen_shannon_divergence(pdf_train, pdf_test),
        "MIA/knn-auroc": roc_auc_score(
            np.concatenate(
                [np.ones_like(train_nearest_dist), np.zeros_like(test_nearest_dist)],
                axis=0,
            ),
            np.concatenate([train_nearest_dist, test_nearest_dist], axis=0),
        ),
    }

    # # plot PDFs
    # fig = go.Figure()
    # _ = fig.add_trace(go.Scatter(x=x_values, y=kde_train(x_values),name="Train"))
    # _ = fig.add_trace(go.Scatter(x=x_values, y=kde_test(x_values),name="Test"))
    # fig.show()

    # # plot normalized histograms
    # fig = go.Figure()
    # _ = fig.add_trace(go.Histogram(x=train_nearest_dist.flatten(), histnorm='probability density',name="Train"))
    # _ = fig.add_trace(go.Histogram(x=test_nearest_dist.flatten(), histnorm='probability density',name="Test"))
    # fig.show()

    return metrics


# @hydra.main(config_path="configs", config_name="config", version_base=None)
def evaluate(inputs, wandb_task_name="DEBUG", config={}):

    # folderName = f"{opt.method}-r{opt.ratio}-{opt.cwgan_path}-{opt.pixgan_path}"

    # Create wandb run to save the results

    state_vars = inputs["state_vars"]
    CORR_METHOD = "ffill"
    CORR_TH = 0.2
    MODEL_NAME = "TimEHR"
    DATASET = "p12"

    if not os.path.exists(f"./Results/{wandb_task_name}/"):
        os.makedirs(f"./Results/{wandb_task_name}/")
    wandb.init(
        config=config,
        project="TimEHR-Eval",
        entity="hokarami",
        name=wandb_task_name,
        reinit=True,
        dir=f"./Results/{wandb_task_name}/TimEHR-Eval",
    )

    df_ts_fake, df_static_fake = inputs["df_ts_fake"], inputs["df_static_fake"]
    df_ts_train, df_static_train = inputs["df_ts_train"], inputs["df_static_train"]
    df_ts_test, df_static_test = inputs["df_ts_test"], inputs["df_static_test"]

    X_train, y_train = xgboost_embeddings(df_ts_train, state_vars)
    X_fake, y_fake = xgboost_embeddings(df_ts_fake, state_vars)

    X_test_by_train, y_test = xgboost_embeddings(
        df_ts_test, state_vars, df_base=df_ts_train
    )
    X_test_by_fake, _ = xgboost_embeddings(df_ts_test, state_vars, df_base=df_ts_fake)

    # compataiblity with the previous version
    df_train_fake, df_static_fake = df_ts_fake, df_static_fake
    df_train_real, df_static_real = df_ts_train, df_static_train
    df_test, df_static = df_ts_test, df_static_test

    X_train_real, y_train_real = X_train, y_train
    X_train_fake, y_train_fake = X_fake, y_fake

    X_test_real, y_test_real = X_test_by_train, y_test
    X_test_fake, y_test_fake = X_test_by_fake, y_test

    # df_train_real = df_train_real.merge(df_static_real[['RecordID','Label']],on=['RecordID'],how='inner')
    # df_train_fake = df_train_fake.merge(df_static_fake[['RecordID','Label']],on=['RecordID'],how='inner')
    # df_test = df_test.merge(df_static[['RecordID','Label']],on=['RecordID'],how='inner')

    # X_train_real, X_train_fake, X_test_real, X_test_fake,\
    # y_train_real, y_train_fake, y_test_real, y_test_fake = xgboost_embeddings(df_train_real, df_train_fake, df_test, state_vars)

    # if opt.privacy_emb=='summary':
    #     LL = 4*len(state_vars)
    #     X_real = pd.concat([X_train_real.fillna(0).iloc[:,:LL], y_train_real], axis=1)
    #     X_fake = pd.concat([X_train_fake.fillna(0).iloc[:,:LL], y_train_fake], axis=1)
    #     X_test = pd.concat([X_test_real.fillna(0).iloc[:,:LL], y_test_real], axis=1)
    # elif opt.privacy_emb=='summary2':
    LL = 5 * len(state_vars)
    X_real = pd.concat([X_train_real.fillna(0).iloc[:, :LL], y_train_real], axis=1)
    X_fake = pd.concat([X_train_fake.fillna(0).iloc[:, :LL], y_train_fake], axis=1)
    X_test = pd.concat([X_test_real.fillna(0).iloc[:, :LL], y_test_real], axis=1)

    # # save some exmaples
    # class Opt:
    #     def __init__(self, NVARS):
    #         self.NVARS = NVARS
    #         self.epoch = -1

    # # Creating an instance of the class with the attribute value set to 21
    # my_object = Opt(NVARS=len(state_vars))
    # print(inputs['train_data'].shape)
    save_examples(
        torch.from_numpy(inputs["train_data"][:10]),
        torch.from_numpy(inputs["fake_data"][:10]),
        n_ts=len(state_vars),
        epoch_no=-1,
    )

    # folderName = "M3-r1.0-['hokarami', 'CWGAN', '2zt63m5f']-['hokarami', 'PIXGAN', 'ov985tal']"
    # X_train_real2 = pd.read_pickle(f".local/_reuse/{folderName}/X_train_real.pkl")
    # X_train_fake2 = pd.read_pickle(f".local/_reuse/{folderName}/X_train_fake.pkl")
    # X_test_real2 = pd.read_pickle(f".local/_reuse/{folderName}/X_test_real.pkl")
    # X_test_fake2 = pd.read_pickle(f".local/_reuse/{folderName}/X_test_fake.pkl")

    # y_train_real2 = pd.read_pickle(f".local/_reuse/{folderName}/y_train_real.pkl")
    # y_train_fake2 = pd.read_pickle(f".local/_reuse/{folderName}/y_train_fake.pkl")
    # y_test_real2 = pd.read_pickle(f".local/_reuse/{folderName}/y_test_real.pkl")
    # y_test_fake2 = pd.read_pickle(f".local/_reuse/{folderName}/y_test_fake.pkl")

    # metrics = compute_utility(X_train_real2, X_train_fake2, X_test_real, X_test_fake2,\
    #         y_train_real, y_train_fake2, y_test_real, y_test_fake2)

    # compute utility
    if DATASET in ["p12", "p19", "mimic"]:
        metrics = compute_utility(
            X_train_real,
            X_train_fake,
            X_test_real,
            X_test_fake,
            y_train_real,
            y_train_fake,
            y_test_real,
            y_test_fake,
        )
        wandb.log(metrics)

    # compute temporal correlation
    if DATASET in ["p12", "p19", "mimic-big"]:
        CORR_METHOD = "ffill"
    else:
        CORR_METHOD = "ffill"
    metric = compute_temp_corr(
        df_train_real,
        df_train_fake,
        df_test,
        state_vars,
        corr_method=CORR_METHOD,
        corr_th=CORR_TH,
    )
    print(metric)
    wandb.log(metric)

    # now resample to 10k
    # N_samples = min(1000000, len(X_test))
    N_samples = min(len(X_fake), len(X_real), len(X_test))
    # N_samples=5000
    print(f"\n[info] resampling to {N_samples}")
    X_real = X_real.sample(N_samples, random_state=42, replace=False)
    X_fake = X_fake.sample(N_samples, random_state=42, replace=False)
    X_test = X_test.sample(N_samples, random_state=42, replace=False)

    # IMPORTANT: we have some variables that are sort of discrete. Hence the histogram of real and test is no longer similar to a continuous variable. Synthcity will somehow detects this fact and will report low scores.
    X_real = X_real + np.random.normal(0, 0.00001, X_real.shape)
    X_fake = X_fake + np.random.normal(0, 0.00001, X_fake.shape)
    X_test = X_test + np.random.normal(0, 0.00001, X_test.shape)

    # SYNTHCITY
    metrics = compute_synthcity(X_real, X_fake, X_test)
    wandb.log(metrics)

    REAL = X_real.drop(columns="outcome").values
    FAKE = X_fake.drop(columns="outcome").values
    TEST = X_test.drop(columns="outcome").values

    # compute privacy NNAA
    metrics_sym, metrics_asym, metrics_asym_bl = compute_nnaa(REAL, FAKE, TEST)
    wandb.log(metrics_sym)
    # wandb.log(metrics_asym)
    # wandb.log(metrics_asym_bl)

    # compute privacy MIA-kNN
    metrics = compute_mia_knn(REAL, FAKE, TEST)
    wandb.log(metrics)

    # plot tsne
    fig_tsne = plot_tsne(REAL, FAKE, N=10000)
    wandb.log({"t-SNE": wandb.Plotly(fig_tsne)})

    wandb.finish()


if __name__ == "__main__":

    evaluate()

    pass
