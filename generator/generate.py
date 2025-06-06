import argparse
from pathlib import Path

import torch
from PIL import Image

def generation(tokenizer: Path, autoregressive: Path, n_images: int, output: Path):
    """
    Tokenize images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    n_images: Number of image to generate
    output: Path to save the images
    """
    output = Path(output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = torch.load(tokenizer, weights_only=False, map_location=torch.device(device)).to(device)
    ar_model = torch.load(autoregressive, weights_only=False, map_location=torch.device(device)).to(device)

    dummy_x, _, _ = tk_model.encode(torch.zeros(1, 3, 128, 128, device=device))
    dummy_index = tk_model.encode_int(dummy_x)
    _, h, w = dummy_index.shape

    generations = ar_model.generate(n_images, h, w, device=device)
    images = tk_model.decode(tk_model.decode_int(generations)).cpu().transpose(1,3)
    np_images = (255 * (images).clip(0, 1)).to(torch.uint8).numpy()
    for idx in range(0, np_images.shape[0]):
        Image.fromarray(np_images[idx,:,:,:], 'RGB').save(output / f"generation_{idx}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, default="BSQTokenizer.th")
    parser.add_argument("--autoregressive", type=str, default="AutoRegressive.th")
    parser.add_argument("--n_images", type=int, default=1)
    parser.add_argument("--output", type=str, default="./generated_imgs/")

    generation(**vars(parser.parse_args()))