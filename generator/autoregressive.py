import torch

class AutoregressiveModel(torch.nn.Module):
    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.embed = torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=n_tokens)
        decoder_layer = torch.nn.TransformerEncoderLayer(d_model=n_tokens, nhead=8, dim_feedforward=d_latent, batch_first=True)
        self.decoder = torch.nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.pad = torch.nn.ConstantPad1d((1,0),0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.to("cuda:0")
        # print(x.shape) # [64, 20, 30]
        # print(x.dtype) # torch.int64
        # print(x.min()) # 0
        # print(x.max()) # 1023
        x = x.flatten(start_dim=1)
        # print(x.shape) # [64, 600]
        x = self.embed(x)
        # print(x.shape) # [64, 600, 1024]
        x = self.pad(x.permute(0,2,1))
        x = x.permute(0,2,1)
        # print(x.shape) # [64, 601, 1024]
        # decoder expects [batch, seq_len, d_model]
        x = self.decoder(x, torch.nn.Transformer.generate_square_subsequent_mask(601, device=self.device), is_causal=True)
        # print(x.shape) # [64, 601, 1024]
        x = x[:,:-1,:] # remove padding
        # print(x.shape) # [64, 600, 1024]
        x = x.contiguous().view(-1, 20, 30, 1024)
        # print(x.shape) # [64, 20, 30, 1024]
        return x

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        y_hat = torch.zeros((B, 1), dtype=torch.long, device=device)
        x = y_hat[:,0:1]
        x = self.embed(x)
        x = self.decoder(x, torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=device), is_causal=True)
        x = torch.nn.functional.softmax(x, dim=2)
        x = x[:,-1,:]
        x = torch.multinomial(x, num_samples=1)
        y_hat = torch.cat([y_hat, x], dim=1)
        for i in range(1, h*w):
            x = y_hat[:,0:(i+1)]
            x = self.embed(x)
            x = self.decoder(x, torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=device), is_causal=True)
            x = torch.nn.functional.softmax(x, dim=2)
            x = x[:,-1,:]
            x = torch.multinomial(x, num_samples=1)
            y_hat = torch.cat([y_hat, x], dim=1)
        y_hat = y_hat[:, 1:]
        y_hat = y_hat.reshape(-1, h, w)
        return y_hat.contiguous()