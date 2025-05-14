# gen-dog

### Warning: This project is still a work in progress...

### Repo Description
This goal of this repository is to create a website similar to [this-person-does-not-exist](this-person-does-not-exist.com), a popular website that generates images of people that do not exist using the StyleGAN model, but instead for images of dogs. My primary motivation for project this was to explore transformer-based image generation models and try to use them for a (somewhat) useful purpose. This project uses a "two-stage" image generation model, where the first stage involves a model to tokenize images into a compressed representation. The second stage is a transformer to learn the tokenized image representations and generate new images! The tokenization model here is based on [Binary Spherical Quantization](https://arxiv.org/abs/2406.07548) but uses CNNs instead of ViTs. The transformer model in stage two can be an autoregressive model or a masked generation model (similar to MaskGIT). Unfortunately, while training an image tokenization model has been relatively trivial, I have so far been unsuccessful in training the generative model (stage two) without having to use very large amounts of compute...

The code is written in python and heavily uses the pytorch library. 

### Data
The data for training the image generation model comes from the [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data) which is ~20k images of dogs seperated into 120 breeds. A cropped (to 128x128) and centered version of this dataset can be found under `./data/` in this repo. 

### Getting Started
Clone the repository:

`git clone link`

Install dependencies:

`pip install -r requirements.txt`

Python >= 3.10 is needed use this repo.

### Using this Repo

`./generator/` contains model and training code for two-stage image generator. 

`./data/` cropped image data.

`./app/` code for image generation wesbite (uses streamlit).

`./logs/` tensorboard logs from various training experiments. 

`./classifier/` code for ConvNext classification model and training. 