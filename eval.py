

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

from Utils.data import load_data, Physio3

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Evaluate the results of the model')
    parser.add_argument('path', type=str, help='Path to the results folder')
    parser.add_argument('--method', type=str, default='ctgan', help='method for generating static data')

    parser.add_argument('--count', type=int, default=0, help='Number of synthetic samples to generate, if 0, generate as many as the samples in training data')
    args = parser.parse_args()

    print(args.path)

    # setup wandb
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_KEY"))

    # load progress.yaml
    progress = OmegaConf.load(args.path + '/progress.yaml')

    # load config.yaml    
    config = OmegaConf.load(args.path + '/config.yaml')

    
    # loading data
    path_train, path_val = config.data.path_train.format(SPLIT=config.split), config.data.path_eval.format(SPLIT=config.split)

    train_dataset, val_dataset, train_schema, val_schema = load_data(path_train, path_val, batch_size = config.bs)


    # initialize the model
    model = TimEHR(config)

    # update the model from the saved weights
    model.from_pretrained(path_cwgan=progress.cwgan_run_path, path_pix2pix=progress.pix2pix_run_path)
    

    # get synthetic data
    counts = args.count if args.count!=0 else len(train_dataset)
    fake_static, fake_data = model.generate(train_dataset, train_schema, count = counts, method=args.method)
    df_ts_fake, df_demo_fake = mat2df(fake_data,fake_static, train_schema)

    # get train data
    train_static, train_data = model._get_data(train_dataset)
    df_ts_train, df_demo_train = mat2df(train_data,train_static, train_schema)
    
    # get test data
    test_static, test_data = model._get_data(val_dataset)
    df_ts_test, df_demo_test = mat2df(test_data,test_static, val_schema)


    # prepare inputs
    inputs = {

        # normalized data
        'fake_static': fake_static,
        'fake_data': fake_data,
        'train_static': train_static,
        'train_data': train_data,
        'test_static': test_static,
        'test_data': test_data,

        # dataframes from normalized data
        'df_ts_fake': df_ts_fake,
        'df_demo_fake': df_demo_fake,
        'df_ts_train': df_ts_train,
        'df_demo_train': df_demo_train,
        'df_ts_test': df_ts_test,
        'df_demo_test': df_demo_test,

        'state_vars': train_schema.dynamic_processor['mean'].index.tolist(),
    }


    # we save the results in a wandb task   
    wandb_task_name = f"{config.data.name}-s{config.split}"
    wandb_config = {'data':config.data.name, 'split':config.split}
    evaluate(inputs, wandb_task_name = wandb_task_name, config=wandb_config)