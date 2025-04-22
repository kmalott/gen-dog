import torch

class MaskedModel(torch.nn.Module):
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
        self.embed = torch.nn.Embedding(num_embeddings=self.n_tokens+1, embedding_dim=d_model)
        decoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_latent, batch_first=True)
        self.decoder = torch.nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.token_head = torch.nn.Linear(d_model, self.n_tokens)
        self.mask_token = 2**codebook

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_masked = x.clone()
        x_masked[mask] = self.mask_token
        # print(x.shape) # [B, seq_len]
        x_masked = self.embed(x_masked)
        # print(x.shape) # [B, seq_len, d_model]
        x_masked = self.pos_encode(x_masked)
        # print(x.shape) # [B, seq_len, d_model]
        # decoder expects [batch, seq_len, d_model]
        x_masked = self.decoder(x_masked)
        # print(x.shape) # [B, seq_len, d_model]
        x_masked = self.token_head(x_masked)
        # print(x.shape) # [B, seq_len, n_tokens]
        return x_masked

    def generate(self, B: int = 1, h: int = 32, w: int = 32, steps: int = 8, device=None) -> torch.Tensor:
        pass
    
