import torch

class AutoregressiveModel(torch.nn.Module):
    class PositionalEncoding2D(torch.nn.Module):
        def __init__(self, height, width, dim):
            super().__init__()
            self.row_embed = torch.nn.Parameter(torch.zeros(1, height, dim // 2))
            self.col_embed = torch.nn.Parameter(torch.zeros(1, width, dim // 2))
            torch.nn.init.trunc_normal_(self.row_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.col_embed, std=0.02)

        def forward(self, x):
            H, W = self.row_embed.shape[1], self.col_embed.shape[1]
            pos = torch.cat([
                self.row_embed.transpose(1, 2).unsqueeze(3).expand(-1, -1, -1, W),  # [1, dim//2, H, W]
                self.col_embed.transpose(1, 2).unsqueeze(2).expand(-1, -1, H, -1),  # [1, dim//2, H, W]
            ], dim=1)
            pos = pos.flatten(2).transpose(1, 2)
            return x + pos

    class PositionalEncoding(torch.nn.Module):
        def __init__(self, seq_len, d_model):
            super().__init__()
            self.pos_embed = torch.nn.Parameter(torch.zeros(1, seq_len, d_model))
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        def forward(self, x):
            return x + self.pos_embed

    def __init__(self, d_latent: int = 1024, d_model: int = 512, codebook: int = 14, nhead: int = 1, num_layers: int = 1):
        super().__init__()
        self.n_tokens = 2**codebook
        self.pos_encode = self.PositionalEncoding2D(32, 32, d_model)
        self.embed = torch.nn.Embedding(num_embeddings=self.n_tokens, embedding_dim=d_model)
        decoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_latent, batch_first=True)
        self.decoder = torch.nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.token_head = torch.nn.Linear(d_model, self.n_tokens)
        self.pad = torch.nn.ConstantPad1d((1, 0), 0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask = self.generate_2d_local_mask(32, 32, 9, causal=True, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.to("cuda:0")
        # print(x.shape) # [B, 32, 32]
        # print(x.dtype) # torch.int64
        # print(x.min()) # 0
        # print(x.max()) # 1023
        # h = x.shape[1]
        # w = x.shape[2]
        # x = x.flatten(start_dim=1)
        seq_len = x.shape[1]
        # print(x.shape) # [B, seq_len]
        x = self.embed(x)
        # print(x.shape) # [B, seq_len, d_model]
        x = self.pos_encode(x)
        # print(x.shape) # [B, seq_len, d_model]
        # decoder expects [batch, seq_len, d_model]
        # x = self.decoder(x, torch.nn.Transformer.generate_square_subsequent_mask((seq_len), device=self.device), is_causal=True)
        x = self.decoder(x, self.mask, is_causal=True)
        # print(x.shape) # [B, seq_len, d_model]
        x = self.token_head(x)
        # print(x.shape) # [B, seq_len, n_tokens]
        # x = x.contiguous().view(-1, h, w, self.n_tokens)
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
    
    def generate_2d_local_mask(self, height, width, window_size, causal=True, device=None):
        seq_len = height * width
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)

        # Flattened index to (row, col)
        def idx_to_coords(idx):
            return divmod(idx, width)

        for i in range(seq_len):
            row_i, col_i = idx_to_coords(i)

            for j in range(seq_len):
                row_j, col_j = idx_to_coords(j)

                # Check if within local window
                if abs(row_i - row_j) <= window_size and abs(col_i - col_j) <= window_size:
                    if not causal or j <= i:
                        mask[i, j] = 0  # allow attention

        return mask
    
def debug_model(batch_size: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # sample_batch = torch.rand(batch_size, 32, 32).to(device)
    sample_batch = torch.randint(0, 1024, (batch_size, 32, 32))

    print(f"Input shape: {sample_batch.shape}")

    model = AutoregressiveModel()
    output = model(sample_batch)

    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()