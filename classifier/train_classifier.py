import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
import torch.utils.tensorboard as tb
from tqdm import tqdm

from .convnext import ConvNext
from .data import load_data

def train(exp_dir: str = "logs",
    model_name: str = "ConvNext",
    num_epoch: int = 5,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    s1: int = 2,
    s2: int = 2,
    s3: int = 2,
    s4: int = 2,
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

    # load custom ConvNext classifeir
    # classifier = ConvNext(blocks_section_1=s1, blocks_section_2=s2, blocks_section_3=s3, blocks_section_4=s4)
    # classifier = classifier.to(device)

    # load pretrained ConvNext classifier
    weights = ConvNeXt_Small_Weights.DEFAULT
    model = convnext_small(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, 120)
    classifier = model.to(device)

    # load data loaders
    train_data, val_data = load_data('./data/', batch_size=batch_size)

    # create loss functions and optimizer
    cse_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=lr)

    # training loop
    global_step = 0
    for epoch in range(num_epoch):
        classifier.train()

        # reset losses
        train_loss = torch.tensor([0.0])
        val_loss = torch.tensor([0.0])
        train_acc = torch.tensor([0.0])
        val_acc = torch.tensor([0.0])

        # reset accuracy counters
        acc = 0
        total = 0

        for img, label in tqdm(train_data):
            img, label = img.to(device), label.to(device)
            out = classifier(img)
            loss = cse_loss(out, label)
            acc += torch.sum(torch.nn.functional.softmax(out, dim=-1).argmax(dim=-1).cpu() == label.cpu())
            total += label.cpu().shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            global_step += 1
        acc = acc.float() / total
        train_acc = acc

        # reset accuracy counters
        acc = 0
        total = 0

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            classifier.eval()
            for img, label in tqdm(val_data):
                img, label = img.to(device), label.to(device)
                out = classifier(img)
                loss = cse_loss(out, label)
                acc += torch.sum(torch.nn.functional.softmax(out, dim=-1).argmax(dim=-1).cpu() == label.cpu())
                total += label.cpu().shape[0]
                val_loss += loss.item()
            acc = acc.float() / total
            val_acc = acc
        
        # log train and validation loss
        logger.add_scalar('train_loss', train_loss, global_step)
        logger.add_scalar('val_loss', val_loss, global_step)
        logger.add_scalar('train_acc', train_acc, global_step)
        logger.add_scalar('val_acc', val_acc, global_step)

        # print losses
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: \n"
            f"train_loss={train_loss} \n"
            f"val_loss={val_loss} \n"
            f"train_acc={train_acc} \n"
            f"val_acc={val_acc} \n"
        )

    # save a copy of model weights in the log directory
    torch.save(classifier, log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--s1", type=int, default=2)
    parser.add_argument("--s2", type=int, default=2)
    parser.add_argument("--s3", type=int, default=4)
    parser.add_argument("--s4", type=int, default=2)

    # pass all arguments to train
    train(**vars(parser.parse_args()))