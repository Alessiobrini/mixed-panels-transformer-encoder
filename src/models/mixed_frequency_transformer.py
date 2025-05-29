import torch
import torch.nn as nn
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt


def get_sinusoidal_encoding(seq_len, d_model):
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class MixedFrequencyTransformer(nn.Module):
    def __init__(
        self,
        freq_vocab_size: int,
        var_vocab_size: int,
        d_freq: int = 4,
        d_var: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 512
    ):
        super().__init__()
        self.d_input = 1 + d_freq + d_var
        self.d_model = d_model

        # Learnable embeddings
        self.freq_embedding = nn.Embedding(freq_vocab_size, d_freq)
        self.var_embedding = nn.Embedding(var_vocab_size, d_var)

        # Input projection
        self.input_proj = nn.Linear(self.d_input, d_model)

        # Sinusoidal positional encoding (fixed)
        pe = get_sinusoidal_encoding(max_len, d_model)
        self.register_buffer("positional_encoding", pe)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction head
        self.prediction_head = nn.Linear(d_model, 1)

    def forward(
        self,
        value: torch.Tensor,     # [B, T]
        var_id: torch.Tensor,    # [B, T]
        freq_id: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        value_unsqueezed = value.unsqueeze(-1)                # [B, T, 1]
        var_emb = self.var_embedding(var_id)                  # [B, T, d_var]
        freq_emb = self.freq_embedding(freq_id)               # [B, T, d_freq]

        z = torch.cat([value_unsqueezed, var_emb, freq_emb], dim=-1)  # [B, T, d_input]
        z_proj = self.input_proj(z)                                   # [B, T, d_model]

        # Add positional encoding
        pos_enc = self.positional_encoding[:z_proj.size(1), :].unsqueeze(0)  # [1, T, d_model]
        z_proj = z_proj + pos_enc

        out = self.transformer_encoder(z_proj)                       # [B, T, d_model]
        pooled = out.mean(dim=1)                                     # [B, d_model]
        pred = self.prediction_head(pooled)                          # [B, 1]
        return pred.squeeze(-1)                                      # [B]


if __name__ == "__main__":
    freq_vocab_size = 3
    var_vocab_size = 5

    model = MixedFrequencyTransformer(
        freq_vocab_size=freq_vocab_size,
        var_vocab_size=var_vocab_size,
        d_freq=4,
        d_var=4,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        max_len=1000
    )

    print("\nModel architecture:\n")
    print(model)

    print("\nParameter summary:\n")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:40} | {tuple(param.shape)} | {param.numel()} params")
            total_params += param.numel()
    print(f"\nTotal trainable parameters: {total_params:,}")




    # Extract and detach positional encodings from the model
    pos_enc = model.positional_encoding.cpu().detach().numpy()  # [max_len, d_model]

    # Plot heatmap for first N positions and dimensions
    N_pos = 100  # first 100 positions
    D_dim = 32    # first 32 dimensions (for visibility)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(pos_enc[:N_pos, :D_dim], aspect='auto', cmap='viridis')
    plt.colorbar(label="Encoding Value")
    plt.xlabel("Encoding Dimension")
    plt.ylabel("Position Index")
    plt.title("Sinusoidal Positional Encodings (Heatmap)")
    
