# theorysim (JBES theory-validation add-on) -- axial-attention model on the wide panel.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""Small axial-attention encoder on the WIDE equal-frequency panel Z in R^{T x N}.

Two attentions, exactly as in the paper (Sec methodology): a temporal operator from the
rows of Z and a cross-sectional operator from the rows of Z^T,

    B   = softmax(Q_tau K_tau^T / sqrt(d)) in R^{T x T},   Q_tau, K_tau = Z W,
    A_z = softmax(Q_z   K_z^T   / sqrt(d)) in R^{N x N},   Q_z,   K_z   = Z^T W,

with the value map W^v = I (paper sets V = Z), so the attended panel is Z-tilde = B Z A_z.
A linear read-out maps Z-tilde to a k-dim per-time latent, and a head forecasts the target
column. With use_nonlinearity=False this is the linear AB1 ablation used to learn the
operators for E1-E3; with True it is the nonlinear model used in E4. Because attention is
axial (over the N columns and over the T rows separately) it costs only O(N^2 + T^2) and
avoids the O((N*T)^2) blow-up of flattening the panel into one token stream.

nhead is fixed to 1 in this pass (the configured size); multi-head averaging is a future
extension.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mixed_frequency_transformer import get_sinusoidal_encoding


class AxialAttentionMPTE(nn.Module):
    def __init__(self, N, T, k=4, d_model=16, nhead=1, num_layers=1,
                 use_nonlinearity=False, target_col=None, use_positional_encoding=True,
                 tie_qk=False, attn_temp=1.0, target_mode="mask"):
        super().__init__()
        if nhead != 1:
            raise NotImplementedError("nhead=1 only in this pass")
        if num_layers != 1:
            raise NotImplementedError("num_layers=1 only in this pass")
        self.N, self.T, self.k = N, T, k
        self.d_model = d_model
        self.use_nonlinearity = use_nonlinearity
        self.use_positional_encoding = use_positional_encoding
        self.target_col = target_col
        # target_mode: how to hide the contemporaneous target from the input.
        #   "mask" -> zero the whole target column (drops its history too).
        #   "lag"  -> replace it with its own 1-step lag (paper-faithful: the target's PAST is
        #             a legitimate predictor, only the current value is withheld).
        self.target_mode = target_mode
        # tie_qk: share the query/key map so attention is a Gram matrix (diagonal-dominant,
        # identity-like) -> bounded op-norm -> Assumption A.7. attn_temp<1 sharpens it toward
        # identity. Both are Option-B levers for getting A.7-satisfying learned operators.
        self.tie_qk = tie_qk
        self.attn_temp = attn_temp

        # Temporal attention: tokens = rows of Z (dim N).
        self.q_tau = nn.Linear(N, d_model, bias=False)
        self.k_tau = self.q_tau if tie_qk else nn.Linear(N, d_model, bias=False)
        # Cross-sectional attention: tokens = rows of Z^T (dim T).
        self.q_z = nn.Linear(T, d_model, bias=False)
        self.k_z = self.q_z if tie_qk else nn.Linear(T, d_model, bias=False)
        # k-dim latent read-out (dim_feedforward = k) and forecast head.
        self.latent_readout = nn.Linear(N, k, bias=True)
        self.prediction_head = nn.Linear(k, 1, bias=True)
        # Decoder for the reconstruction (autoencoder) objective (Option B): k -> N.
        self.decoder = nn.Linear(k, N, bias=True)

        if use_positional_encoding:
            # Sinusoidal positional encoding on the time axis, shape (T, N).
            self.register_buffer("pe", get_sinusoidal_encoding(T, N), persistent=False)
        else:
            self.pe = None

    def _operators(self, Z):
        """Return (A_z (N x N), B (T x T)), both row-stochastic softmax attentions."""
        scale = (self.d_model ** 0.5) * self.attn_temp        # temp<1 sharpens the softmax
        # Temporal B from rows of Z.
        Qt, Kt = self.q_tau(Z), self.k_tau(Z)                 # (T, d)
        B = torch.softmax(Qt @ Kt.transpose(-1, -2) / scale, dim=-1)   # (T, T)
        # Cross-sectional A_z from rows of Z^T.
        Zt = Z.transpose(-1, -2)                              # (N, T)
        Qz, Kz = self.q_z(Zt), self.k_z(Zt)                  # (N, d)
        A_z = torch.softmax(Qz @ Kz.transpose(-1, -2) / scale, dim=-1) # (N, N)
        return A_z, B

    def _hide_target(self, Z):
        """Hide the contemporaneous target from the input per target_mode (mask or lag)."""
        if self.target_col is None:
            return Z
        Z = Z.clone()
        tc = self.target_col
        if self.target_mode == "lag":
            col = Z[:, tc].clone()
            Z[1:, tc] = col[:-1]                              # input row t sees target_{t-1}
            Z[0, tc] = 0.0
        else:
            Z[:, tc] = 0.0                                    # zero the whole target column
        return Z

    def forward(self, Z, return_latent=False, return_attention=False):
        """Z: (T, N) wide panel. Returns the target-column forecast (T,)."""
        Z = self._hide_target(Z)
        Z_in = Z + self.pe if self.pe is not None else Z

        A_z, B = self._operators(Z_in)
        Ztilde = B @ Z_in @ A_z                               # (T, N), V = I

        latent = self.latent_readout(Ztilde)                  # (T, k)
        if self.use_nonlinearity:
            latent = torch.tanh(latent)
        pred = self.prediction_head(latent).squeeze(-1)       # (T,)

        out = [pred]
        if return_latent:
            out.append(latent)
        if return_attention:
            out.append((A_z, B))
        return out[0] if len(out) == 1 else tuple(out)

    def reconstruct(self, Z):
        """Autoencoder forward (Option B): map the panel through the attention + k-bottleneck
        and decode back to (T, N). No target masking (we reconstruct every column)."""
        Z_in = Z + self.pe if self.pe is not None else Z
        A_z, B = self._operators(Z_in)
        Ztilde = B @ Z_in @ A_z
        latent = self.latent_readout(Ztilde)
        if self.use_nonlinearity:
            latent = torch.tanh(latent)
        return self.decoder(latent)                           # (T, N)

    @torch.no_grad()
    def attention_matrices(self, Z):
        """Convenience: return (A_z, B) for a panel without tracking gradients."""
        Z = self._hide_target(Z)
        Z_in = Z + self.pe if self.pe is not None else Z
        return self._operators(Z_in)


def fit_forecast(model, Z, target, epochs=200, lr=1e-3, weight_decay=0.0, verbose=False):
    """Train the model by MSE on the target column. Z, target are torch tensors.

    Returns the trained model and the final training loss.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    model.train()
    last = float("nan")
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(Z)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
        last = float(loss.item())
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}  mse={last:.6f}")
    model.eval()
    return model, last


def fit_reconstruct(model, Z, epochs=800, lr=1e-2, weight_decay=0.0, verbose=False):
    """Train the model as a linear autoencoder: MSE between reconstruct(Z) and Z (Option B).

    The reconstruction objective forces every column's information through the bottleneck,
    which (Prop 3: linear autoencoder ~ PCA) should pull the attention toward a well-spread,
    identity-like operator that satisfies Assumption A.7, rather than the hub-concentrated
    operator the masked-target forecast objective learns.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    model.train()
    last = float("nan")
    for ep in range(epochs):
        opt.zero_grad()
        rec = model.reconstruct(Z)
        loss = loss_fn(rec, Z)
        loss.backward()
        opt.step()
        last = float(loss.item())
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}  mse={last:.6f}")
    model.eval()
    return model, last
