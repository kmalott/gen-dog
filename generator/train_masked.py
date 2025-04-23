import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn.functional as F

from tqdm import tqdm
import transformers

from .masked import MaskedModel
from .data import TokenDataset

def train(exp_dir: str = "logs",
    model_name: str = "Masked",
    num_epoch: int = 5,
    lr: float = 1e-3,
    batch_size: int = 1024,
    seed: int = 2024,
    codebook: int = 14,
    latent: int = 512,
    d_model: int = 256,
    nhead: int = 2,
    nlayer: int = 2,
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

    masked = MaskedModel(latent, d_model, codebook, nhead, nlayer)
    masked.to(device)
    # load data loaders
    train_token = TokenDataset("train")
    train_token = torch.utils.data.Subset(train_token, indices=list(range(0,1000)))
    val_token = TokenDataset("val")
    train_data = torch.utils.data.DataLoader(train_token, batch_size=batch_size, num_workers=4, shuffle=True)
    val_data = torch.utils.data.DataLoader(val_token, batch_size=batch_size, num_workers=4, shuffle=False)
    # create optimizer
    optimizer = torch.optim.AdamW(params=masked.parameters(), lr=lr)
    # scheduler = transformers.get_cosine_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps=len(train_data)*3,
    #     num_training_steps=len(train_data)*num_epoch
    # )

    global_step = 0
    # training loop
    for epoch in range(num_epoch):
        global_step = 0
        masked.train()
        # reset losses
        train_loss = torch.tensor([0.0])
        train_acc = torch.tensor([0.0])
        train_total = torch.tensor([0.0])
        val_loss = torch.tensor([0.0])
        val_acc = torch.tensor([0.0])
        val_total = torch.tensor([0.0])
        for x in tqdm(train_data):
            x = x.squeeze(1).to(device)
            x = x.flatten(start_dim=1)
            B, S = x.shape
            ratio = torch.tensor([0.1], device=device)
            # ratio = torch.rand(1, device=device)
            # while ratio > 0.5:
            #     ratio = torch.rand(1, device=device)
            mask = torch.rand((1, S), device=device) < ratio
            total = (mask.sum() * B)
            attn_mask = torch.where(mask, float('-inf'), 0.0).repeat(S, 1)
            target = x.clone()
            logits = masked(x, attn_mask)
            loss = F.cross_entropy(logits[mask], target[mask])
            acc = (torch.sum(logits[mask].argmax(dim=1) == target[mask])).cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            train_loss += loss.item()
            train_acc += acc
            train_total += total
            global_step += 1
        train_loss /= train_total
        train_acc /= train_total

        # disable gradient computation and switch to evaluation mode
        # with torch.inference_mode():
        #     masked.eval()
        #     for x in tqdm(val_data):
        #         x = x.squeeze(1).to(device)
        #         x = x.flatten(start_dim=1)
        #         B, S = x.shape
        #         ratio = torch.tensor([0.1], device=device)
        #             # ratio = torch.rand(1, device=device)
        #             # while ratio > 0.5:
        #             #     ratio = torch.rand(1, device=device)
        #         mask = torch.rand((B, S), device=device) < ratio
        #         total = mask.sum()
        #         attn_mask = torch.where(mask, float('-inf'), 0.0).repeat(S, 1)
        #         target = x.clone()
        #         logits = masked(x, attn_mask)
        #         loss = F.cross_entropy(logits[mask], target[mask])
        #         acc = (torch.sum(logits[mask].argmax(dim=1) == target[mask])).cpu()
        #         val_acc += acc
        #         val_total += total
        #         val_loss += loss.item()
        #     # val_loss /= val_total
        #     val_acc /= val_total

        # log average train and val accuracy to tensorboard
        logger.add_scalar('train_loss', train_loss, global_step)
        logger.add_scalar('val_loss', val_loss, global_step)

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: \n"
            f"train_loss={train_loss} \n"
            f"val_loss={val_loss} \n"
            f"train_acc={train_acc} \n" 
            f"val_acc={val_acc}"
        )

    # save a copy of model weights in the log directory
    torch.save(masked, log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--codebook", type=int, default=14)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--nlayer", type=int, default=2)

    # pass all arguments to train
    train(**vars(parser.parse_args()))