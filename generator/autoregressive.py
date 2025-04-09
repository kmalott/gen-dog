import torch

class AutoregressiveModel(torch.nn.Module):
    def __init__(self, d_latent: int = 1024, codebook: int = 14, nhead: int = 8, ):
        super().__init__()
        self.n_tokens = 2**codebook
        self.embed = torch.nn.Embedding(num_embeddings=self.n_tokens, embedding_dim=self.n_tokens)
        decoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.n_tokens, nhead=nhead, dim_feedforward=d_latent, batch_first=True)
        self.decoder = torch.nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.pad = torch.nn.ConstantPad1d((1, 0), 0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.to("cuda:0")
        # print(x.shape) # [B, 32, 32]
        # print(x.dtype) # torch.int64
        # print(x.min()) # 0
        # print(x.max()) # 1023
        h = x.shape[1]
        w = x.shape[2]
        x = x.flatten(start_dim=1)
        seq_len = x.shape[1]
        # print(x.shape) # [B, seq_len]
        x = self.embed(x)
        # print(x.shape) # [B, seq_len, n_tokens]
        x = self.pad(x.permute(0,2,1))
        x = x.permute(0,2,1)
        # print(x.shape) # [B, seq_len, n_tokens]
        # decoder expects [batch, seq_len, d_model]
        x = self.decoder(x, torch.nn.Transformer.generate_square_subsequent_mask((seq_len+1), device=self.device), is_causal=True)
        # print(x.shape) # [B, seq_len+1, n_tokens]
        x = x[:,:-1,:] # remove padding
        # print(x.shape) # [B, seq_len, n_tokens]
        x = x.contiguous().view(-1, h, w, self.n_tokens)
        # print(x.shape) # [B, 32, 32, n_tokens]
        return x

    def generate(self, B: int = 1, h: int = 32, w: int = 32, device=None) -> torch.Tensor:  # noqa
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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