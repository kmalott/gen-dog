import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn.functional as F

from tqdm import tqdm
import torchvision
from lpips import LPIPS

from .tokenizer import BSQTokenizer
from .discriminator import Discriminator
from .data import load_data_loader, load_data

def calc_gradient_penalty(netD, real_data, fake_data, device):
    alpha = torch.rand(real_data.shape[0], 1, 1, 1)
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train(exp_dir: str = "logs",
    model_name: str = "BSQTokenizer",
    num_epoch: int = 5,
    lr: float = 1e-3,
    batch_size: int = 1024,
    seed: int = 2024,
    gamma0: float = 1.0,
    gamma: float = 1.0,
    codebook: int = 14,
    latent: int = 64,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    tokenizer = BSQTokenizer(latent_dim=latent, codebook=codebook, gamma0=gamma0, gamma=gamma)
    tokenizer = tokenizer.to(device)
    # discriminator = Discriminator()
    # discriminator = discriminator.to(device)

    # load data loaders
    train_data, val_data = load_data('./data/', batch_size=batch_size)

    # create loss functions and optimizer
    mse_loss = torch.nn.MSELoss()
    lpips_loss = LPIPS(net='vgg')
    lpips_loss = lpips_loss.to(device)
    optimizer_t = torch.optim.AdamW(params=tokenizer.parameters(), lr=lr)
    # optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), lr=0.0001, betas=(0.0, 0.9))


    global_step = 0
    # metrics = {"train_loss": [], "val_loss": [], 
    #            "train_gan": [], "val_gan": [], 
    #            "train_mse": [], "val_mse": [], 
    #            "train_lpips": [], "val_lpips": [],
    #            "train_entropy": [], "val_entropy": [],
    #            "train_disc": [], "val_disc": []
    #            }
    
    # warmup loop
    warmup = False
    if warmup:
        i = 0
        tokenizer.train()
        for img, label in tqdm(train_data):
            img, label = img.to(device), label.to(device)
            img_hat, _, _ = tokenizer(img)
            mse = mse_loss(img_hat, img)
            lpips = lpips_loss(img_hat, img)
            total_loss_t = 10*mse + 0.5*lpips.sum()
            optimizer_t.zero_grad()
            total_loss_t.backward()
            optimizer_t.step()
            i += 1

        # log imgs after warmup
        grid = torchvision.utils.make_grid(img)
        logger.add_image('images', grid, global_step)
        grid = torchvision.utils.make_grid(img_hat)
        logger.add_image('images_reconstructed', grid, global_step)
    
    # initialize weighed averages (alpha) for LeCAM reg. 
    # alpha_f = 100
    # alpha_r = 100

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        # for key in metrics:
        #     metrics[key].clear()

        tokenizer.train()
        # discriminator.train()

        # reset losses
        train_loss = torch.tensor([0.0])
        val_loss = torch.tensor([0.0])
        train_gan = torch.tensor([0.0])
        val_gan = torch.tensor([0.0])
        train_mse = torch.tensor([0.0])
        val_mse = torch.tensor([0.0])
        train_lpips = torch.tensor([0.0])
        val_lpips = torch.tensor([0.0])
        train_disc = torch.tensor([0.0])
        val_disc = torch.tensor([0.0])
        train_entropy = torch.tensor([0.0])
        val_entropy = torch.tensor([0.0])

        train_fake = torch.tensor([0.0])
        train_real = torch.tensor([0.0])
        train_gp = torch.tensor([0.0])

        i = 0
        for img, label in tqdm(train_data):
            img, label = img.to(device), label.to(device)
            # train discriminator
            # img_hat, _, _ = tokenizer(img).detach()

            # WGAN Loss
            # gan_fake = discriminator(img_hat).mean()
            # gan_real = discriminator(img).mean()
            # gp = calc_gradient_penalty(discriminator, img, img_hat, device)
            # total_loss_d = gan_fake - gan_real + (10*gp) 
            # optimizer_d.zero_grad()
            # total_loss_d.backward()
            # optimizer_d.step()

            # LeCAM Reg. Loss
            # # dis_fake = discriminator(img_hat)
            # # dis_real = discriminator(img)
            # # gan_fake = F.relu(1. + dis_fake).mean()
            # # gan_real = F.relu(1. - dis_real).mean()
            # # alpha_f = (0.99*alpha_f) + (0.01*dis_fake.mean().item())
            # # alpha_r = (0.99*alpha_r) + (0.01*dis_real.mean().item())
            # # reg = reg = torch.mean(F.relu(dis_real - alpha_f).pow(2)) + torch.mean(F.relu(alpha_r - dis_fake).pow(2))
            # # total_loss_d = (0.3*reg) - (gan_real + gan_fake)
            # optimizer_d.zero_grad()
            # total_loss_d.backward()
            # optimizer_d.step()

            # if i % 5 == 0:
            # train tokenizer (generator)
            img_hat, entropy, _ = tokenizer(img)
            mse = mse_loss(img_hat, img)
            lpips = lpips_loss(img_hat, img)
            # gan = discriminator(img_hat).mean()
            # total_loss_t = (10*mse) - (0.1*gan) + (0.5*lpips.sum())
            total_loss_t = (10*mse) + (0.5*lpips.sum()) + (0.1*entropy)
            optimizer_t.zero_grad()
            total_loss_t.backward()
            optimizer_t.step()
            train_loss += total_loss_t.item()
            train_mse += mse.item() * 10
            train_lpips += lpips.sum().item() * 0.5
            train_entropy += entropy.item()
            # train_gan += gan.item() * 0.1

            # train_disc += total_loss_d.item()
            # train_fake += gan_fake.item()
            # train_real += gan_real.item()
            # train_gp += gp.item()
            
            global_step += 1
            i += 1
        # metrics["train_loss"].append(train_loss)
        # metrics["train_gan"].append(train_gan)
        # metrics["train_mse"].append(train_mse)
        # metrics["train_lpips"].append(train_lpips)
        # metrics["train_entropy"].append(train_entropy)
        # metrics["train_disc"].append(train_disc)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            tokenizer.eval()
            # discriminator.eval()
            cb = torch.tensor([], device=device)

            for img, label in tqdm(val_data):
                img, label = img.to(device), label.to(device)
                # validate discriminator
                img_hat, entropy, cb_usage = tokenizer(img)
                # gan_fake = discriminator(img_hat).mean()
                # gan_real = discriminator(img).mean()
                # total_loss_d = gan_fake - gan_real

                # validate tokenizer (generator)
                mse = mse_loss(img_hat, img)
                lpips = lpips_loss(img_hat, img)
                # total_loss_t = (10*mse) - (0.1*gan_fake) + (0.5*lpips.sum())
                total_loss_t = (10*mse) + (0.5*lpips.sum()) + (0.1*entropy)

                # store losses
                val_loss += total_loss_t.item()
                val_mse += mse.item() * 10
                val_lpips += lpips.sum().item() * 0.5
                val_entropy += entropy.item()
                # val_gan += gan_fake.item() * 0.1
                # val_disc += total_loss_d.item()

                # track codebook
                cb = torch.concat((cb, cb_usage))
            # metrics["val_loss"].append(val_loss)
            # metrics["val_gan"].append(val_gan)
            # metrics["val_mse"].append(val_mse)
            # metrics["val_lpips"].append(val_lpips)
            # metrics["val_entropy"].append(val_entropy)
            # metrics["val_disc"].append(val_disc)

        # check codebook utilization
        cb = cb.unique()
        print(f"Codebook unqiue values: {cb.shape[0]} / {(2**codebook)}")
        print(f"Codebook utilization: {cb.shape[0] / (2**codebook)}")

        # log average train and val accuracy to tensorboard
        # epoch_train_loss = torch.as_tensor(metrics["train_loss"])
        # epoch_val_loss = torch.as_tensor(metrics["val_loss"])
        logger.add_scalar('train_loss', train_loss, global_step)
        logger.add_scalar('val_loss', val_loss, global_step)

        # add last of the reconstructed images to tensorboard
        grid = torchvision.utils.make_grid(img)
        logger.add_image('images', grid, global_step)
        grid = torchvision.utils.make_grid(img_hat)
        logger.add_image('images_reconstructed', grid, global_step)

        # print on first, last, every 10th epoch
        #if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: \n"
            f"train_loss={train_loss} \n"
            f"val_loss={val_loss} \n"
            f"train_mse={train_mse} \n"
            f"val_mse={val_mse} \n"
            f"train_lpips={train_lpips} \n"
            f"val_lpips={val_lpips} \n"
            f"train_entropy={train_entropy} \n"
            f"val_entropy={val_entropy} \n"
            f"train_gan={train_gan} \n"
            f"val_gan={val_gan} \n"
            f"train_disc={train_disc} \n"
            f"val_disc={val_disc} \n"
        )
        # print(
        #     f"Real Img Loss: {train_real} \n"
        #     f"Fake Img Loss: {train_fake} \n"
        #     f"Gradient Penalty: {train_gp}"
        # )

    # save a copy of model weights in the log directory
    torch.save(tokenizer.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--gamma0", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--codebook", type=int, default=14)
    parser.add_argument("--latent", type=int, default=64)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
