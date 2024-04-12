
import hydra
from omegaconf import DictConfig, OmegaConf
# from hydra import initialize, compose

import os
import wandb
from dotenv import load_dotenv


from Utils.data import load_data, Physio3

from Utils.models import TimEHR

from Utils.utils import mat2df


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):

    wandb_task_name = f'{cfg.data.name}-s{cfg.split}'
    # saveing cfg to yaml in Results/wandb_task_name
    os.makedirs('Results/'+wandb_task_name, exist_ok=True)
    OmegaConf.save(config=cfg, resolve=True, f='Results/'+wandb_task_name+'/config.yaml')



    # loading data
    path_train, path_val = cfg.data.path_train.format(SPLIT=cfg.split), cfg.data.path_eval.format(SPLIT=cfg.split)

    train_dataset, val_dataset, train_schema, val_schema = load_data(path_train, path_val, batch_size = cfg.bs)


    # training model
    model = TimEHR(cfg)
    
    model.train(train_dataset, val_dataset, wandb_task_name=wandb_task_name)
    model.save_to_yaml(folder=f'Results/{wandb_task_name}')

    # # Alternatively, you can load a pre-trained model
    # # model.from_pretrained(path_cwgan='hokarami/CWGAN/uj5gf643',path_pix2pix='hokarami/PIXGAN/lorajx1i')

    # # get synthetic data
    # fake_static, fake_data = model.generate(train_dataset, train_schema)
    # df_ts_fake, df_demo_fake = mat2df(fake_data,fake_static, train_schema)

    # print(df_ts_fake)

    pass


if __name__=="__main__":

    # configurations are managed by hydra. You can modify them in the config/config.yaml file.

    # setup wandb
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_KEY"))

    # api = wandb.Api()

    train()
