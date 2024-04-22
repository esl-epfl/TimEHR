# Raw Datasets


## P12
- download the dataset from [PhysioNet/Computing in Cardiology Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/). The dataset contains three folder for three hospitals. You can also download the csv file containing all 12k patients from [here](https://drive.google.com/drive/folders/112_jRjKB8_oFlyF8J0eS9xqhDxBLfaPh?usp=drive_link)(`p12.csv`).

## P19
- download the dataset from [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/). The dataset contains two folders for two hospitals. You can also download the csv file containing all ~43k patients from [here](https://drive.google.com/drive/folders/112_jRjKB8_oFlyF8J0eS9xqhDxBLfaPh?usp=drive_link)(`df_A.csv` and `df_B.csv`).

## MIMIC-III
- You need to request access to the dataset from [PhysioNet](https://physionet.org/content/mimiciii/1.4/). Once you downloaded the csv files, you can use this repo to extract ~50k patients into a `.csv` fuke.

Put the csv files in the `data/raw` folder or another folder. Then, add the `path_raw` to the `configs/data/{DATASET_NAME}.yaml` file.

## Simulated data
- Please run: 
    ```bash
    python gen_sim.py --n-vars 16 32 64 128 --lambdas 0.2 0.5 1 2
    python gen_sim.py --n-vars 16 --lambdas 0.5
    ```
    This will create 16 simulated datasets with different number of features and lambdas. The raw and processed datasets will be saved in the `data/raw` and `data/processed` folders, respectively. Create yaml file for the simulated data in the `configs/data/{DATASET_NAME}.yaml`.


# Preparing Datasets
* Now, check [Prepare_Datasets.ipynb](Prepare_Datasets.ipynb) to see how to prepare the raw datasets.


