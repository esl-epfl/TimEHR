# stdlib
import logging
import os
import typing
from typing import Dict, Optional

# third party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# from utils import
import torch.optim as optim
import wandb
import yaml
from Utils.modules import Critic, Disc_pix, Gen_pix, Generator1, initialize_weights
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# from utils import mat2df, prepro,
from Utils.utils import ffill, find_last_epoch, gradient_penalty, save_examples

# synthcity absolute
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


class CWGAN(nn.Module):
    def __init__(self, opt: OmegaConf) -> None:
        super(CWGAN, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.gen = Generator1(
            self.opt.z_dim,
            self.opt.channels,
            self.opt.img_size,
            self.opt.feat_gen,
            d_conditional=self.opt.d_conditional,
            conditional=self.opt.cond,
            kernel_size=self.opt.kernel_size,
        ).to(self.device)

        self.critic = Critic(
            self.opt.channels,
            self.opt.img_size,
            self.opt.feat_cri,
            d_conditional=self.opt.d_conditional,
            conditional=self.opt.cond,
            kernel_size=self.opt.kernel_size,
        ).to(self.device)

        initialize_weights(self.gen)
        initialize_weights(self.critic)

        # gen.train()
        # critic.train()

        # print number of params in each module
        print(
            f"Generator has {sum(p.numel() for p in self.gen.parameters() if p.requires_grad):,} trainable parameters"
        )
        print(
            f"Critic has {sum(p.numel() for p in self.critic.parameters() if p.requires_grad):,} trainable parameters"
        )
        # initializate optimizer
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=opt.lr, betas=(0.0, 0.9))
        self.opt_critic = optim.Adam(
            self.critic.parameters(), lr=opt.lr, betas=(0.0, 0.9)
        )

        self.run_path = "CWGAN initialized"
        self.epoch_no = 0

    def from_pretrained(self, run_path: str) -> None:
        logging.info(f"Loading CWGAN from run: {run_path}")
        print("CWGAN - run_path: ", run_path)
        api = wandb.Api()
        run = api.run(run_path)

        for file in run.files():
            if ".pt" in file.name:
                _ = file.download(
                    replace=False, exist_ok=True, root=f".local/{run_path}"
                )

        run.config
        if "d_static" in run.config.keys():
            run.config["d_conditional"] = run.config["d_static"]

        # stdlib
        from datetime import datetime

        run_creation_time = datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%S")
        ref_time = datetime(2024, 4, 4)
        if run_creation_time < ref_time:
            print("run is old")
            Z_DIM, N_CHANNELS, FEATURES_GEN, FEATURES_CRITIC = (
                run.config["Z_DIM"],
                run.config["CHANNELS_IMG"],
                run.config["FEATURES_GEN"],
                run.config["FEATURES_CRITIC"],
            )
            IMG_SIZE, d_conditional, COND, KERNEL_SIZE = (
                run.config["IMG_SIZE"],
                run.config["D_STATIC"],
                run.config["COND"],
                run.config["KERNEL_SIZE"],
            )
        else:
            Z_DIM, N_CHANNELS, FEATURES_GEN, FEATURES_CRITIC = (
                run.config["z_dim"],
                run.config["channels"],
                run.config["feat_gen"],
                run.config["feat_cri"],
            )
            IMG_SIZE, d_conditional, COND, KERNEL_SIZE = (
                run.config["img_size"],
                self.opt.d_conditional,
                run.config["cond"],
                run.config["kernel_size"],
            )
            pass

        # COND = run.config['COND']
        # d_conditional = run.config['d_conditional']

        last_epoch = find_last_epoch(f".local/{run_path}/model_weights")
        # last_epoch = 150
        print("last_epoch: ", last_epoch)

        checkpoint = torch.load(f".local/{run_path}/model_weights/gen_{last_epoch}.pt")
        self.gen = Generator1(
            Z_DIM,
            N_CHANNELS,
            IMG_SIZE,
            FEATURES_GEN,
            d_conditional,
            conditional=COND,
            kernel_size=KERNEL_SIZE,
        ).to("cuda")
        _ = self.gen.load_state_dict(checkpoint, strict=False)

        checkpoint = torch.load(
            f".local/{run_path}/model_weights/critic_{last_epoch}.pt"
        )
        self.critic = Critic(
            N_CHANNELS,
            IMG_SIZE,
            FEATURES_CRITIC,
            d_conditional,
            conditional=COND,
            kernel_size=KERNEL_SIZE,
        ).to("cuda")
        _ = self.critic.load_state_dict(checkpoint, strict=False)

        # # load optimizer
        # checkpoint = torch.load(f'.local/{run_path}/model_weights/opt_gen_{last_epoch}.pt')
        # _ = self.opt_gen.load_state_dict(checkpoint)

        # checkpoint = torch.load(f'.local/{run_path}/model_weights/opt_critic_{last_epoch}.pt')
        # _ = self.opt_critic.load_state_dict(checkpoint)

        self.run_path = run_path
        self.epoch_no = last_epoch

    def eval_epoch(self, loader: DataLoader) -> tuple:

        self.gen.eval()
        self.critic.eval()

        sample_real = []
        sample_fake = []

        tot_loss_critic = 0
        tot_loss_gen = 0
        tot_n_samples = 0

        # Target labels not needed! <3 unsupervised
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, leave=False)):

                mask = batch[0].to(self.device)
                value = batch[1].to(self.device)
                cur_batch_size = mask.shape[0]

                sta = batch[2].to(self.device) if self.opt.cond else None

                # if self.opt.channels ==2:
                real = torch.cat([value, mask], dim=1)  # shape is (bs,2,64,64)
                # else:
                #     real = mask
                y = sta

                # if sample_real is None:

                # sample noise
                noise = torch.randn(cur_batch_size, self.opt.z_dim, 1, 1).to(
                    self.device
                )

                # generate a batch of images from noise and conditional info
                fake = self.gen(noise, y=y)  # bs,ch,H,W   bs,1,64,64

                # compute CRITIC scores for real and fake images
                critic_real = self.critic(real, y=y).reshape(-1)  # bs
                critic_fake = self.critic(fake, y=y).reshape(-1)  # bs
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))

                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))

                gen_fake = self.critic(fake, y=y).reshape(-1)
                loss_gen = -torch.mean(gen_fake)

                tot_loss_critic += loss_critic.item() * cur_batch_size
                tot_loss_gen += loss_gen.item() * cur_batch_size
                # tot_loss_reg += loss_reg.item()
                tot_n_samples += cur_batch_size

                sample_real.append(real)
                sample_fake.append(fake)

        wandb.log(
            {
                "Eval/loss_critic": tot_loss_critic / tot_n_samples,
                "Eval/loss_gen": tot_loss_gen / tot_n_samples,
            },
            step=self.epoch_no,
            commit=False,
        )

        sample_real = torch.cat(sample_real, dim=0)
        sample_fake = torch.cat(sample_fake, dim=0)

        return sample_real, sample_fake

    def train_epoch(self, train_loader: DataLoader) -> None:

        self.gen.train()
        self.critic.train()
        # if disc_pred is not None:
        #     disc_pred.train()
        # Target labels not needed! <3 unsupervised
        tot_loss_critic = 0
        tot_loss_gen = 0
        tot_loss_reg = 0
        tot_n_samples = 1e-3

        tot_L1 = 0
        tot_corr_loss = 0
        for batch_idx, batch in enumerate(
            tqdm(train_loader, leave=False, desc="Batches")
        ):
            img_size = batch[0].shape[-1]
            mask = batch[0].to(self.device)
            value = batch[1].to(self.device)
            cur_batch_size = mask.shape[0]

            sta = batch[2].to(self.device) if self.opt.cond else None
            if self.opt.channels == 2:
                real = torch.cat([value, mask], dim=1)  # shape is (bs,2,64,64)
            else:
                real = mask
            y = sta

            for _ in range(self.opt.cri_iter):

                # sample noise
                noise = torch.randn(cur_batch_size, self.opt.z_dim, 1, 1).to(
                    self.device
                )

                # generate a batch of images from noise and conditional info
                fake = self.gen(noise, y=y)  # bs,ch,H,W   bs,1,64,64

                # compute CRITIC scores for real and fake images
                critic_real = self.critic(real, y=y).reshape(-1)  # bs
                critic_fake = self.critic(fake, y=y).reshape(-1)  # bs
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))

                # compute the gradient penalty
                gp = gradient_penalty(self.critic, real, fake, y=y, device=self.device)
                loss_reg = self.opt.lambda_gp * gp

                # total critic loss
                loss_critic_reg = loss_critic + loss_reg

                # backprop and optimize
                self.critic.zero_grad()
                loss_critic_reg.backward(retain_graph=True)
                self.opt_critic.step()

            # compute L1 loss
            L1 = nn.L1Loss()(fake, real)

            # compute correlation loss
            NN = fake[:, 0].reshape(-1, img_size).shape[0]
            CORR_fake = (
                torch.matmul(
                    fake[:, 0].reshape(-1, img_size).T, fake[:, 0].reshape(-1, img_size)
                )
                / NN
            )
            CORR_real = (
                torch.matmul(
                    real[:, 0].reshape(-1, img_size).T, real[:, 0].reshape(-1, img_size)
                )
                / NN
            )

            CORR_LOSS = nn.L1Loss()(CORR_fake, CORR_real)

            gen_fake = self.critic(fake, y=y).reshape(-1)

            # compute total loss for generator
            loss_gen = (
                -torch.mean(gen_fake)
                + L1 * self.opt.lambda_l1
                + CORR_LOSS * self.opt.lambda_corr
            )

            # backprop and optimize
            self.gen.zero_grad()
            loss_gen.backward()
            self.opt_gen.step()

            tot_loss_critic += loss_critic.item() * cur_batch_size / cur_batch_size
            tot_loss_gen += loss_gen.item() * cur_batch_size / cur_batch_size
            tot_loss_reg += loss_reg.item() * cur_batch_size / cur_batch_size
            # tot_loss_pred += pred_loss.item()
            tot_n_samples += cur_batch_size
            tot_L1 += L1.item() * cur_batch_size / cur_batch_size
            tot_corr_loss += CORR_LOSS.item() * cur_batch_size / cur_batch_size

        wandb.log(
            {
                "Train/loss_reg": tot_loss_reg / tot_n_samples,
                "Train/loss_critic": tot_loss_critic / tot_n_samples,
                "Train/loss_gen": tot_loss_gen / tot_n_samples,
                # "Train/loss_pred": tot_loss_pred/tot_n_samples,
                "Train/L1": tot_L1 / tot_n_samples,
                "Train/corr_loss": tot_corr_loss / tot_n_samples,
            },
            step=self.epoch_no,
            commit=False,
        )
        # Print losses occasionally and print to tensorboard

        pass

    def train(
        self,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        wandb_task_name: str = "DEBUG",
        collate_fn: Optional[typing.Callable] = None,
    ) -> None:
        # N_VARS = len(train_dataset.dynamic_processor['mean'])
        N_VARS = len(train_dataset.temporal_features)

        # creating dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.opt.bs,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
            # num_workers=8
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.opt.bs,
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
            # num_workers=8
        )

        print("### Start Training CWGAN-GP ...")
        wandb_config = OmegaConf.to_container(
            self.opt, resolve=True, throw_on_missing=True
        )

        if not os.path.exists(f"Results/{wandb_task_name}/CWGAN"):
            os.makedirs(f"Results/{wandb_task_name}/CWGAN")

        wandb.init(
            config=wandb_config,
            project=self.opt.wandb_project_name,
            entity="hokarami",
            name=wandb_task_name,
            reinit=True,
            dir=f"Results/{wandb_task_name}/CWGAN",
        )

        self.run_path = wandb.run.path

        for epoch in tqdm(range(self.opt.epochs + 1), desc="Epochs"):
            self.epoch_no = epoch

            if epoch > 0:
                # train for one epoch
                self.train_epoch(train_loader)

            # # evaluate train data
            # _, _ = self.eval_epoch(train_loader,prefix='Eval_Train')

            # log some examples
            if (epoch) % 5 == 0:
                # evaluate val data
                sample_real, sample_fake = self.eval_epoch(val_loader)
                save_examples(
                    sample_real[:10],
                    sample_fake[:10],
                    n_ts=N_VARS,
                    epoch_no=self.epoch_no,
                )

            # log model weights
            if (epoch) % 50 == 0:
                # create temp directory if it does not exist
                if not os.path.exists("./model_weights"):
                    os.makedirs("./model_weights")
                torch.save(self.gen.state_dict(), f"./model_weights/gen_{epoch}.pt")
                torch.save(
                    self.critic.state_dict(), f"./model_weights/critic_{epoch}.pt"
                )

                # saving optimizer and scheduler
                torch.save(
                    self.opt_gen.state_dict(), f"./model_weights/opt_gen_{epoch}.pt"
                )
                torch.save(
                    self.opt_critic.state_dict(),
                    f"./model_weights/opt_critic_{epoch}.pt",
                )

                wandb.save(f"model_weights/gen_{epoch}.pt")
                wandb.save(f"model_weights/critic_{epoch}.pt")
                wandb.save(f"model_weights/opt_gen_{epoch}.pt")
                wandb.save(f"model_weights/opt_critic_{epoch}.pt")

            wandb.log({"dummy": 0})

        wandb.finish()


class Pix2Pix(nn.Module):
    def __init__(self, opt: OmegaConf) -> None:
        super(Pix2Pix, self).__init__()
        self.opt = opt
        self.device = opt.device

        self.gen = Gen_pix(
            opt.channels * 0 + 1,
            opt.img_size,
            features=opt.feat_gen,
            d_conditional=self.opt.d_conditional,
            conditional=opt.cond,
            kernel_size=opt.kernel_size,
        ).to(self.device)

        self.disc = Disc_pix(
            opt.channels * 0 + 2,
            opt.img_size,
            features=list(opt.feat_disc),
            d_conditional=self.opt.d_conditional,
            conditional=opt.cond,
            kernel_size=opt.kernel_size,
        ).to(self.device)

        # compute and print number of parameters
        num_params_gen = sum(
            p.numel() for p in self.gen.parameters() if p.requires_grad
        )
        num_params_disc = sum(
            p.numel() for p in self.disc.parameters() if p.requires_grad
        )
        print(f"Generator has {num_params_gen} parameters")
        print(f"Discriminator has {num_params_disc} parameters")

        # initialize parameters
        initialize_weights(self.gen)
        initialize_weights(self.disc)

        self.opt_disc = optim.Adam(
            self.disc.parameters(),
            lr=opt.lr,
            betas=(0.5, 0.999),
        )
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

        self.run_path = "Pix2Pix initialized"
        self.epoch_no = 0

    def from_pretrained(self, run_path: str) -> None:
        logging.info(f"Loading PIXGAN from run: {run_path}")
        print("PIXGAN - run_path: ", run_path)
        api = wandb.Api()
        run = api.run(run_path)

        for file in run.files():
            if ".pt" in file.name:
                _ = file.download(
                    replace=False, exist_ok=True, root=f".local/{run_path}"
                )

        if "d_static" in run.config.keys():
            run.config["d_conditional"] = run.config["d_static"]

        last_epoch = find_last_epoch(f".local/{run_path}/model_weights")
        print("last_epoch: ", last_epoch)

        # stdlib
        from datetime import datetime

        run_creation_time = datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%S")
        ref_time = datetime(2024, 4, 4)
        if run_creation_time < ref_time:
            print("run is old")
            N_CHANNELS, FEAT_GEN, FEAT_DISC = (
                run.config["CHANNELS_IMG"],
                run.config["FEAT_GEN"],
                [8, 16],
            )
            IMG_SIZE, d_conditional, COND, KERNEL_SIZE = (
                run.config["IMG_SIZE"],
                run.config["d_conditional"],
                run.config["COND"],
                run.config["KERNEL_SIZE"],
            )
        else:
            N_CHANNELS, FEAT_GEN, FEAT_DISC = (
                run.config["channels"],
                run.config["feat_gen"],
                run.config["feat_disc"],
            )
            IMG_SIZE, d_conditional, COND, KERNEL_SIZE = (
                run.config["img_size"],
                self.opt.d_conditional,
                run.config["cond"],
                run.config["kernel_size"],
            )
            pass

        checkpoint = torch.load(f".local/{run_path}/model_weights/gen_{last_epoch}.pt")

        self.gen = Gen_pix(
            N_CHANNELS * 0 + 1,
            IMG_SIZE,
            features=FEAT_GEN,
            d_conditional=d_conditional,
            conditional=COND,
            kernel_size=KERNEL_SIZE,
        ).to(self.device)
        _ = self.gen.load_state_dict(checkpoint, strict=False)

        checkpoint = torch.load(f".local/{run_path}/model_weights/disc_{last_epoch}.pt")
        self.disc = Disc_pix(
            N_CHANNELS * 0 + 2,
            IMG_SIZE,
            features=FEAT_DISC,
            d_conditional=d_conditional,
            conditional=COND,
            kernel_size=KERNEL_SIZE,
        ).to(self.device)
        _ = self.disc.load_state_dict(checkpoint, strict=False)

        # _ = self.gen.eval()
        # _ = self.disc.eval()

        self.run_path = run_path
        self.epoch_no = last_epoch

    def eval_epoch(self, loader: DataLoader) -> tuple:

        self.disc.eval()
        self.gen.eval()

        sample_real = []
        sample_fake = []

        loop = tqdm(loader, leave=False)

        tot_L1 = 0
        tot_L2 = 0
        tot_G_fake_loss = 0
        tot_D_loss = 0
        tot_samples = 0
        tot_corr_loss = 0

        for idx, batch in enumerate(loop):
            x = batch[0].to(self.device)  # (batch_size, 3, 256, 256)
            y = batch[1].to(self.device)  # (batch_size, 3, 256, 256)
            sta = batch[2].to(self.device) if self.opt.cond else None

            # Train Discriminator
            with torch.no_grad():
                img_size = x.shape[2]
                # y_partial = y*0
                # y_fake = gen(x,y_partial, sta=sta)
                y_fake = self.gen(x, sta=sta)
                # sta = gen.embed(sta).detach()
                y_fake = y_fake.masked_fill(x < 0, 0)

                D_real = self.disc(x, y, sta=sta)
                D_real_loss = nn.BCEWithLogitsLoss()(D_real, torch.ones_like(D_real))
                D_fake = self.disc(x, y_fake.detach(), sta=sta)
                D_fake_loss = nn.BCEWithLogitsLoss()(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

                G_fake_loss = nn.BCEWithLogitsLoss()(D_fake, torch.ones_like(D_fake))
                L1 = nn.L1Loss()(y_fake, y)  # * opt.L1_LAMBDA
                L2 = nn.MSELoss()(y_fake, y)  # * opt.L2_LAMBDA

                # compute correlation loss
                NN = y_fake.reshape(-1, img_size).shape[0]
                CORR_fake = (
                    torch.matmul(
                        y_fake.reshape(-1, img_size).T, y_fake.reshape(-1, img_size)
                    )
                    / NN
                )
                CORR_real = (
                    torch.matmul(y.reshape(-1, img_size).T, y.reshape(-1, img_size))
                    / NN
                )
                CORR_LOSS = nn.L1Loss()(CORR_fake, CORR_real)

                # G_loss = G_fake_loss + L1

            if idx % 10 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                )
            tot_D_loss += D_loss.item()
            tot_G_fake_loss += G_fake_loss.item()
            tot_L1 += L1.item()
            tot_L2 += L2.item()
            # tot_pred_loss += pred_loss.item()
            tot_samples += x.size(0)
            tot_corr_loss += CORR_LOSS.item()

            real = torch.cat([y, x], dim=1)  # shape is (bs,2,64,64)
            fake = torch.cat([y_fake, x], dim=1)  # shape is (bs,2,64,64)

            sample_real.append(real)
            sample_fake.append(fake)

        wandb.log(
            {
                "Eval/D_loss": tot_D_loss / tot_samples * self.opt.bs,
                "Eval/G_fake_loss": tot_G_fake_loss / tot_samples * self.opt.bs,
                "Eval/L1_loss": tot_L1 / tot_samples * self.opt.bs,
                "Eval/L2_loss": tot_L2 / tot_samples * self.opt.bs,
                "Eval/corr_loss": tot_corr_loss / tot_samples * self.opt.bs,
                # "Eval/pred_loss": tot_pred_loss / tot_samples*opt.BATCH_SIZE,
                # "Eval/AUROC": metric_auroc,
                # "Eval/AUPRC": metric_auprc,
                # "Eval/F1": metric_f1,
            },
            step=self.epoch_no,
            commit=False,
        )

        sample_real = torch.cat(sample_real, dim=0)
        sample_fake = torch.cat(sample_fake, dim=0)

        return sample_real, sample_fake

    def train_epoch(self, loader: DataLoader) -> None:

        self.disc.train()
        self.gen.train()

        # loop =

        tot_L1_filled = 0

        tot_G_fake_loss = 0
        tot_D_loss = 0
        tot_samples = 0
        tot_loss_reg = 0

        tot_L1 = 0
        tot_L2 = 0
        tot_corr_loss = 0
        for idx, batch in enumerate(tqdm(loader, leave=False, desc="Batches")):
            x = batch[0].to(self.device)  # (batch_size, 3, 256, 256)
            y = batch[1].to(self.device)  # (batch_size, 3, 256, 256)
            sta = batch[2].to(self.device) if self.opt.cond else None

            # Train Discriminator
            for _ in range(self.opt.disc_iter):
                # with torch.cuda.amp.autocast():
                # y_partial = make_partial(y, x)
                # y_fake = gen(x, y_partial, sta=sta)
                y_fake = self.gen(x, sta=sta)
                # sta2 = gen.embed(sta).detach()
                # sta2 = self.disc.embed(sta)
                y_fake = y_fake.masked_fill(x < 0, 0)
                D_real = self.disc(x, y, sta=sta)
                D_real_loss = nn.BCEWithLogitsLoss()(D_real, torch.ones_like(D_real))
                # D_real_loss = -torch.mean(D_real)
                D_fake = self.disc(x, y_fake.detach(), sta=sta)
                D_fake_loss = nn.BCEWithLogitsLoss()(D_fake, torch.zeros_like(D_fake))
                # D_fake_loss = torch.mean(D_fake)
                D_loss = (D_real_loss + D_fake_loss) / 2

                # # compute the gradient penalty
                # gp = gradient_penalty2(disc, x,y, y_fake,sta=sta, device=self.device)
                # loss_reg = gp * 10
                # D_loss = D_loss + loss_reg

                # backprop and optimize
                self.disc.zero_grad()
                self.d_scaler.scale(D_loss).backward()
                self.d_scaler.step(self.opt_disc)
                self.d_scaler.update()

            # Train generator
            with torch.cuda.amp.autocast():
                img_size = x.shape[2]
                y_fake = self.gen(x, sta=sta)
                y_fake = y_fake.masked_fill(x < 0, 0)

                D_fake = self.disc(x, y_fake, sta=sta)
                G_fake_loss = nn.BCEWithLogitsLoss()(D_fake, torch.ones_like(D_fake))

                # compute L1 Loss
                if self.opt.limit > 0:
                    y_filled = ffill(y.squeeze(), x.squeeze())
                    y_fake_filled = ffill(y_fake.squeeze(), x.squeeze())

                L1_filled = nn.L1Loss()(y_fake_filled, y_filled)  # * self.opt.
                L1 = nn.L1Loss()(y_fake, y)
                L2 = nn.MSELoss()(y_fake, y)
                # L2_filled = nn.MSELoss()(y_fake_filled, y_filled)

                # calculate L1 when 50% of x>0 is maskes

                # L11 = nn.L1Loss()(y_fake, y.masked_fill(x<0,0))
                # L1_weighted = ((y_fake-y).abs().mean(0)*self.opt.w).sum()

                # compute correlation loss
                NN = y_fake.reshape(-1, img_size).shape[0]
                CORR_fake = (
                    torch.matmul(
                        y_fake.reshape(-1, img_size).T, y_fake.reshape(-1, img_size)
                    )
                    / NN
                )
                CORR_real = (
                    torch.matmul(y.reshape(-1, img_size).T, y.reshape(-1, img_size))
                    / NN
                )
                CORR_LOSS = nn.L1Loss()(CORR_fake, CORR_real)

                # compute total loss for generator
                G_loss = (
                    G_fake_loss
                    + L2 * self.opt.lambda_l1
                    + CORR_LOSS * self.opt.lambda_corr
                )

            # backprop and optimize
            self.opt_gen.zero_grad()
            self.g_scaler.scale(G_loss).backward()
            self.g_scaler.step(self.opt_gen)
            self.g_scaler.update()

            # if idx % 10 == 0:
            #     loop.set_postfix(
            #         D_real=torch.sigmoid(D_real).mean().item(),
            #         D_fake=torch.sigmoid(D_fake).mean().item(),
            #     )
            tot_D_loss += D_loss.item()
            tot_G_fake_loss += G_fake_loss.item()
            tot_L1 += L1.item()
            tot_L2 += L2.item()
            tot_L1_filled += L1_filled.item()
            # tot_L1_weighted += L1_weighted.item()
            tot_samples += x.size(0)
            # tot_pred_loss += pred_loss.item()
            tot_corr_loss += CORR_LOSS.item()
            # tot_loss_reg += loss_reg.item()

        wandb.log(
            {
                "Train/D_loss": tot_D_loss / tot_samples * self.opt.bs,
                "Train/G_fake_loss": tot_G_fake_loss / tot_samples * self.opt.bs,
                "Train/L1_loss": tot_L1 / tot_samples * self.opt.bs,
                "Train/L2_loss": tot_L2 / tot_samples * self.opt.bs,
                "Train/L1_filled_loss": tot_L1_filled / tot_samples * self.opt.bs,
                # "Train/L1_weighted_loss": tot_L1_weighted / tot_samples*self.opt.bs,
                # "Train/pred_loss": tot_pred_loss / tot_samples*self.opt.bs,
                "Train/corr_loss": tot_corr_loss / tot_samples * self.opt.bs,
                "Train/loss_reg": tot_loss_reg / tot_samples * self.opt.bs,
            },
            step=self.epoch_no,
            commit=False,
        )

    def train(
        self,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        wandb_task_name: str = "DEBUG",
        collate_fn: Optional[typing.Callable] = None,
    ) -> None:
        # N_VARS = len(train_dataset.dynamic_processor['mean'])
        N_VARS = len(train_dataset.temporal_features)
        # creating dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.opt.bs,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.opt.bs,
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
        )
        print("### Start Training Pix2Pix ...")
        wandb_config = OmegaConf.to_container(
            self.opt, resolve=True, throw_on_missing=True
        )

        if not os.path.exists(f"Results/{wandb_task_name}/Pix2Pix"):
            os.makedirs(f"Results/{wandb_task_name}/Pix2Pix")
        wandb.init(
            config=wandb_config,
            project=self.opt.wandb_project_name,
            entity="hokarami",
            name=wandb_task_name,
            reinit=True,
            dir=f"Results/{wandb_task_name}/Pix2Pix",
        )

        self.run_path = wandb.run.path

        for epoch in tqdm(range(self.opt.epochs + 1), desc="Epochs"):
            self.epoch_no = epoch

            if epoch > 0:
                # train for one epoch
                self.train_epoch(train_loader)

            # log some examples
            if (epoch) % 5 == 0:

                #     # eval train data
                #     sample_real, sample_fake = eval_fn(disc, gen, train_loader, opt,prefix='Eval_Train')

                # eval val data
                sample_real, sample_fake = self.eval_epoch(val_loader)

                save_examples(
                    sample_real[:10],
                    sample_fake[:10],
                    n_ts=N_VARS,
                    epoch_no=self.epoch_no,
                )

            # log model weights
            if (epoch) % 50 == 0:
                # create temp directory if it does not exist
                if not os.path.exists("./model_weights"):
                    os.makedirs("./model_weights")
                torch.save(self.gen.state_dict(), f"./model_weights/gen_{epoch}.pt")
                torch.save(self.disc.state_dict(), f"./model_weights/disc_{epoch}.pt")

                # saving optimizer and scalars
                torch.save(
                    self.opt_gen.state_dict(), f"./model_weights/opt_gen_{epoch}.pt"
                )
                torch.save(
                    self.opt_disc.state_dict(), f"./model_weights/opt_disc_{epoch}.pt"
                )

                torch.save(
                    self.g_scaler.state_dict(), f"./model_weights/g_scaler_{epoch}.pt"
                )
                torch.save(
                    self.d_scaler.state_dict(), f"./model_weights/d_scaler_{epoch}.pt"
                )

                wandb.save(f"model_weights/gen_{epoch}.pt")
                wandb.save(f"model_weights/disc_{epoch}.pt")
                wandb.save(f"model_weights/opt_gen_{epoch}.pt")
                wandb.save(f"model_weights/opt_disc_{epoch}.pt")
                wandb.save(f"model_weights/g_scaler_{epoch}.pt")
                wandb.save(f"model_weights/d_scaler_{epoch}.pt")

            wandb.log({"dummy": 0})

        wandb.finish()


class TimEHR(nn.Module):
    def __init__(self, opt: OmegaConf) -> None:
        super(TimEHR, self).__init__()
        self.opt = opt
        self.device = opt.device

        self.progress: Dict = dict()  # to store progress of training

        self.cwgan = CWGAN(opt.cwgan)
        self.pix2pix = Pix2Pix(opt.pix2pix)

    def from_pretrained(self, path_cwgan: str = "", path_pix2pix: str = "") -> None:
        if path_cwgan != "":
            self.cwgan.from_pretrained(path_cwgan)
        if path_pix2pix != "":
            self.pix2pix.from_pretrained(path_pix2pix)

        pass

    def train(
        self,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        wandb_task_name: str = "DEBUG",
        collate_fn: Optional[typing.Callable] = None,
    ) -> None:

        self.cwgan.train(
            train_dataset,
            val_dataset,
            wandb_task_name=wandb_task_name,
            collate_fn=collate_fn,
        )
        self.pix2pix.train(
            train_dataset,
            val_dataset,
            wandb_task_name=wandb_task_name,
            collate_fn=collate_fn,
        )

    def generate(
        self,
        train_dataset: DataLoader,
        collate_fn: Optional[typing.Callable] = None,
        count: int = 1000,
        method: str = "ctgan",
    ) -> tuple:

        # if self.opt.generation.ctgan:
        if method == "ctgan":
            logging.info("Generating static data using TabularGAN")

            # import sys
            # sys.path.insert(0, "/mlodata1/hokarami/synthcity/src")

            real_static, real_data = self._get_data(
                train_dataset, collate_fn=collate_fn
            )
            X = pd.DataFrame(
                real_static, columns=train_dataset.static_features + ["outcome"]
            )
            loader = GenericDataLoader(
                X,
                target_column="outcome",
                # sensitive_columns=["sex"],
            )

            syn_model = Plugins().get("ctgan", n_iter=100)

            syn_model.fit(loader, cond=X["outcome"])

            # generate
            # count = int(real_static.shape[0]*self.opt.generation.ratio)

            real_prevalence = X["outcome"].mean()
            # generate a random binary vector with exact same prevalence as real data
            conditional = np.random.binomial(1, real_prevalence, size=(count,))

            generated_conditional = (
                syn_model.generate(
                    count=count,
                    cond=conditional,
                )
                .dataframe()
                .values
            )  # generate only patients with the outcome = 1
            print(f"{count} samples generated using CTGAN")
        elif method == "real":
            logging.info(
                f"Using static data from the training set ({len(train_dataset)} samples)"
            )
            real_static, real_data = self._get_data(
                train_dataset, collate_fn=collate_fn
            )
            generated_conditional = real_static
        else:
            raise ValueError(f"Unknown method for generation: {method}")

        IMG_SIZE = self.opt.cwgan.img_size
        Z_DIM = self.opt.cwgan.z_dim
        # METHOD = self.opt.generation.method
        all_fake = []
        all_fake_sta = []
        while generated_conditional.shape[0] > 0:
            cur_batch_size = min(generated_conditional.shape[0], IMG_SIZE)
            sta_fake = (
                torch.from_numpy(generated_conditional[:cur_batch_size])
                .to(self.device)
                .float()
            )
            generated_conditional = generated_conditional[cur_batch_size:]

            # sta_fake = sta
            all_fake_sta.append(sta_fake.cpu().detach().numpy())

            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(self.device)

            fake = self.cwgan.gen(noise, y=sta_fake)  # bs,ch,H,W   64,1,64,64
            x_fake = fake[:, -1, :, :].unsqueeze(1)
            # select 0 as threshold and set above to 1 and below to -1
            x_fake = torch.where(x_fake > 0, 1, -1)

            # PIXGAN

            x_pixgan = x_fake

            y_fake = self.pix2pix.gen(x_pixgan, sta=sta_fake)
            y_fake = y_fake.masked_fill(x_pixgan < 0, 0)
            fake = torch.cat([y_fake, x_pixgan], axis=1)

            all_fake.append(fake.cpu().detach().numpy())

        all_fake = np.concatenate(all_fake, axis=0)
        all_fake_sta = np.concatenate(all_fake_sta, axis=0)

        # df_train_fake,  df_demo_fake = mat2df(all_fake,all_fake_sta, train_schema)
        # df_train_fake = df_train_fake.merge(df_demo_fake[['RecordID','Label']],on=['RecordID'],how='inner')

        # df_train_real, df_demo_real = mat2df(real_data,real_static, train_schema)
        # # df_train_real = df_train_real.merge(df_demo_real[['RecordID','Label']],on=['RecordID'],how='inner')

        # output = {
        #     'df_train_fake': df_train_fake,
        #     'df_demo_fake': df_demo_fake,
        #     'all_fake': all_fake,
        #     'all_fake_sta': all_fake_sta,
        #     'df_train_real': df_train_real,
        #     'df_demo_real': df_demo_real,
        #     'all_real': real_data,
        #     'all_real_sta': real_static,
        # }

        return all_fake_sta, all_fake

        pass

    def _get_data(
        self, dataset: DataLoader, collate_fn: Optional[typing.Callable] = None
    ) -> tuple:

        loader = DataLoader(
            dataset,
            batch_size=self.opt.bs,
            collate_fn=dataset.collate_fn,
            shuffle=False,
        )

        all_sta = []
        all_real = []
        for batch_idx, batch in enumerate(tqdm(loader)):

            mask = batch[0].to(self.device)
            value = batch[1].to(self.device)
            # cur_batch_size = x.shape[0]

            sta = batch[2].to(self.device)
            real = torch.cat([value, mask], dim=1)  # shape is (bs,2,64,64)
            # y=sta
            # y_partial = y*0
            # y_fake = gen(x,y_partial, sta=sta)
            all_sta.append(sta.cpu().detach().numpy())
            all_real.append(real.cpu().detach().numpy())

        all_sta = np.concatenate(all_sta, axis=0)
        all_real = np.concatenate(all_real, axis=0)
        return all_sta, all_real

    def save_to_yaml(self, folder: str = "Results") -> None:

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.progress.update(
            {
                "cwgan_run_path": self.cwgan.run_path,
                "pix2pix_run_path": self.pix2pix.run_path,
                "cwgan_epoch_no": self.cwgan.epoch_no,
                "pix2pix_epoch_no": self.pix2pix.epoch_no,
            }
        )

        # write to yaml
        with open(f"{folder}/progress.yaml", "w") as file:
            yaml.dump(self.progress, file)

        pass
