import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from .data import TokenDataset

def test_ar_forward(tokenizer, autoregressive):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = torch.load(tokenizer, weights_only=False, map_location=torch.device(device)).to(device)
    ar_model = torch.load(autoregressive, weights_only=False, map_location=torch.device(device)).to(device)

    val_data = torch.utils.data.DataLoader(TokenDataset("train"), batch_size=1, num_workers=4, shuffle=False)

    for x in tqdm(val_data):
        x = x.squeeze(1).to(device)
        print(x.shape) # [1, 32, 32]
        x_hat = ar_model(x)
        print(x_hat.shape) # [1, 32, 32, 16384]
        x_hat = x_hat.squeeze().view(-1, 2**14).cpu()
        print(x_hat.shape) # [1024, 16384]
        x_hat = torch.nan_to_num(x_hat, 0.0)
        x_hat = x_hat.clip(0, 2**14)
        x_hat = torch.multinomial(x_hat, num_samples=1)
        print(x_hat.shape) # [1024, 1]
        x_hat = x_hat.view(-1, 32, 32).to(device)
        print(x_hat.shape) # [1, 32, 32]
        break

    # save results as imgs
    images = tk_model.decode(tk_model.decode_int(x)).cpu().transpose(1,3)
    np_images = (255 * (images).clip(0, 1)).to(torch.uint8).numpy()
    for idx in range(0, np_images.shape[0]):
        Image.fromarray(np_images[idx,:,:,:], 'RGB').save("original.png")

    images = tk_model.decode(tk_model.decode_int(x_hat)).cpu().transpose(1,3)
    np_images = (255 * (images).clip(0, 1)).to(torch.uint8).numpy()
    for idx in range(0, np_images.shape[0]):
        Image.fromarray(np_images[idx,:,:,:], 'RGB').save("reconstructed.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="BSQTokenizer.th")
    parser.add_argument("--autoregressive", type=str, default="AutoRegressive.th")
    test_ar_forward(**vars(parser.parse_args()))