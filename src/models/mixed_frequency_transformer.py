import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MixedFrequencyTransformer(nn.Module):
    def __init__(
        self,
        freq_vocab_size: int,
        time_vocab_size: int,
        var_vocab_size: int,
        d_value: int = 8,
        d_freq: int = 4,
        d_time: int = 8,
        d_var: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_input = d_value + d_freq + d_time + d_var
        self.d_model = d_model

        # Learnable embeddings
        self.freq_embedding = nn.Embedding(freq_vocab_size, d_freq)
        self.time_embedding = nn.Embedding(time_vocab_size, d_time)
        self.var_embedding = nn.Embedding(var_vocab_size, d_var)

        # Project scalar value
        self.value_proj = nn.Linear(1, d_value)

        # Input projection to model dimension
        self.input_proj = nn.Linear(self.d_input, d_model)

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
        time_id: torch.Tensor    # [B, T]
    ) -> torch.Tensor:
        value_proj = self.value_proj(value.unsqueeze(-1))  # [B, T, d_value]
        var_emb = self.var_embedding(var_id)               # [B, T, d_var]
        freq_emb = self.freq_embedding(freq_id)            # [B, T, d_freq]
        time_emb = self.time_embedding(time_id)            # [B, T, d_time]
    
        z = torch.cat([value_proj, var_emb, freq_emb, time_emb], dim=-1)  # [B, T, d_input]
        z_proj = self.input_proj(z)                                       # [B, T, d_model]
        out = self.transformer_encoder(z_proj)                            # [B, T, d_model]
        pooled = out.mean(dim=1)                                          # [B, d_model]
        pred = self.prediction_head(pooled)                               # [B, 1]
        return pred.squeeze(-1)                                           # [B]
    

