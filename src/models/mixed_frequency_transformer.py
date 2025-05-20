import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MixedFrequencyTransformer(nn.Module):
    def __init__(
        self,
        raw_input_dim: int,
        freq_vocab_size: int,
        time_vocab_size: int,
        d_freq: int = 4,
        d_time: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_input = raw_input_dim + d_freq + d_time
        self.d_model = d_model

        # Frequency and time embeddings
        self.freq_embedding = nn.Embedding(freq_vocab_size, d_freq)
        self.time_embedding = nn.Embedding(time_vocab_size, d_time)

        # Linear projection to model dimension
        self.input_proj = nn.Linear(self.d_input, d_model)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.prediction_head = nn.Linear(d_model, 1)

    def forward(
        self,
        raw_input: torch.Tensor,     # [B, D]
        freq_id: torch.Tensor,       # [B]
        time_id: torch.Tensor        # [B]
    ) -> torch.Tensor:
        # Embeddings
        freq_emb = self.freq_embedding(freq_id)   # [B, d_freq]
        time_emb = self.time_embedding(time_id)   # [B, d_time]

        # Concatenate all
        z = torch.cat([raw_input, freq_emb, time_emb], dim=-1)  # [B, d_input]
        z_proj = self.input_proj(z).unsqueeze(0)  # [1, B, d_model] if single sequence

        # Transformer expects sequence of shape [batch, seq, d_model] if batch_first=True
        out = self.transformer_encoder(z_proj)  # [1, B, d_model]

        # Predict from each time step
        pred = self.prediction_head(out.squeeze(0))  # [B, 1]
        return pred.squeeze(-1)  # [B]
