from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _resolve_activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation '{name}'")


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
        use_nonlinearity: bool,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation_fn = activation_fn
        self.use_nonlinearity = use_nonlinearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.linear1(out)
        if self.use_nonlinearity:
            out = self.activation_fn(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return residual + out


class FeedForwardStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
        use_nonlinearity: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FeedForwardBlock(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    use_nonlinearity=use_nonlinearity,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x)
        return x


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
        max_len: int = 512,
        dim_feedforward: int = 2048,
        activation: str = "relu",
        use_nonlinearity: bool = True,
        use_attention: bool = True,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.d_input = 1 + d_freq + d_var
        self.d_model = d_model
        self.use_attention = use_attention
        self.max_len = max_len
        self.positional_encoding_enabled = use_positional_encoding

        # Learnable embeddings
        self.freq_embedding = nn.Embedding(freq_vocab_size, d_freq)
        self.var_embedding = nn.Embedding(var_vocab_size, d_var)

        # Input projection
        self.input_proj = nn.Linear(self.d_input, d_model)

        if self.positional_encoding_enabled:
            pe = get_sinusoidal_encoding(max_len, d_model)
            self.register_buffer("positional_encoding", pe)
        else:
            self.register_buffer("positional_encoding", None)

        # Transformer encoder
        activation_fn = _resolve_activation_fn(activation)
        encoder_activation = activation_fn if use_nonlinearity else _identity

        if use_attention:
            encoder_layer = TransformerEncoderLayer(
                                                    d_model=d_model,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout,
                                                    activation=encoder_activation,
                                                    batch_first=True
                                                )

            self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.transformer_encoder = FeedForwardStack(
                num_layers=num_layers,
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation_fn=activation_fn,
                use_nonlinearity=use_nonlinearity,
            )

        # Prediction head
        self.prediction_head = nn.Linear(d_model, 1)
        
        # Learnable scaling parameter for positional encoding
        # self.positional_scale = nn.Parameter(torch.tensor(1.0))
        
        self.z_proj_norm = nn.LayerNorm(d_model)
        self.pos_enc_norm = nn.LayerNorm(d_model)



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

        z_proj = self.z_proj_norm(z_proj)

        if self.positional_encoding_enabled:
            pos_enc = self._get_positional_encoding(z_proj)
            pos_enc = self.pos_enc_norm(pos_enc)
            z_proj = z_proj + pos_enc

        out = self.transformer_encoder(z_proj)                       # [B, T, d_model]
        pooled = out.mean(dim=1)                                     # [B, d_model]
        pred = self.prediction_head(pooled)                          # [B, 1]
        return pred.squeeze(-1)                                      # [B]

    def _get_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )

        if self.positional_encoding is None:
            raise RuntimeError(
                "Positional encoding is disabled but _get_positional_encoding was called"
            )

        return self.positional_encoding[:seq_len, :].unsqueeze(0)



if __name__ == "__main__":
    freq_vocab_size = 3
    var_vocab_size = 5

    model = MixedFrequencyTransformer(
        freq_vocab_size=freq_vocab_size,
        var_vocab_size=var_vocab_size,
        d_freq=4,
        d_var=4,
        d_model=32,
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
    if model.positional_encoding is not None:
        pos_enc = model.positional_encoding.cpu().detach().numpy()  # [max_len, d_model]
    else:
        raise RuntimeError("No positional encoding available to visualize")

    # Plot heatmap for first N positions and dimensions
    N_pos = 100  # first 100 positions
    D_dim = 32    # first 32 dimensions (for visibility)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(pos_enc[:N_pos, :D_dim], aspect='auto', cmap='viridis')
    plt.colorbar(label="Encoding Value")
    plt.xlabel("Encoding Dimension")
    plt.ylabel("Position Index")
    plt.title("Sinusoidal Positional Encodings (Heatmap)")
    
