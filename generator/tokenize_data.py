import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .data import load_data

def tokenize(tokenizer: str, 
             output_train: str, 
             output_val: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.load(Path(tokenizer), weights_only=False).to(device)

    output_train = Path(output_train)
    output_val = Path(output_val)

    train_data, val_data = load_data('./data/', batch_size=1, save_output=False)

    compressed_imgs_train = []
    compressed_imgs_val = []
    with torch.inference_mode():
            model.eval()
            for img, label in tqdm(train_data):
                img, label = img.to(device), label.to(device)
                img_token = model.encode_int(img)
                compressed_imgs_train.append(img_token)

            for img, label in tqdm(val_data):
                img, label = img.to(device), label.to(device)
                img_token = model.encode_int(img)
                compressed_imgs_val.append(img_token)

    imgs_train = torch.stack(compressed_imgs_train)
    imgs_val = torch.stack(compressed_imgs_val)


    np_compressed_tensor = imgs_train.numpy()
    if np_compressed_tensor.max() < 2**8:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint8)
    elif np_compressed_tensor.max() < 2**16:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint16)
    else:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint32)
    torch.save(np_compressed_tensor, output_train)

    np_compressed_tensor = imgs_val.numpy()
    if np_compressed_tensor.max() < 2**8:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint8)
    elif np_compressed_tensor.max() < 2**16:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint16)
    else:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint32)
    torch.save(np_compressed_tensor, output_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenzier", type=str, default="BSQTokenzier.th")
    parser.add_argument("--output_train", type=str, default="tokenized_data/train_tokenized.pth")
    parser.add_argument("--output_val", type=str, default="tokenized_data/val_tokenized.pth")

    tokenize(**vars(parser.parse_args()))
