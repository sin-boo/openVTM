"""Pose adapter: maps 16 Live2D/face params -> cross-attention tokens for SD 1.5."""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseAdapter(nn.Module):
    """Embed a fixed-length float param vector as cross-attention tokens.

    Same idea as ParamEmbedder in the DiT stack: per-param value MLP + ID embed,
    producing (B, num_params, cross_attention_dim) tokens that are concatenated
    onto the text encoder hidden states before the UNet forward.
    """

    def __init__(
        self,
        num_params: int = 16,
        cross_attention_dim: int = 768,
        mlp_hidden: int = 512,
        num_tokens_per_param: int = 1,
        drop_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_params = int(num_params)
        self.cross_attention_dim = int(cross_attention_dim)
        self.num_tokens_per_param = int(num_tokens_per_param)
        self.drop_prob = float(drop_prob)
        token_dim = cross_attention_dim

        self.value_mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, token_dim * self.num_tokens_per_param),
        )
        self.param_id_embed = nn.Embedding(
            self.num_params, token_dim * self.num_tokens_per_param
        )
        n_tokens = self.num_params * self.num_tokens_per_param
        self.learnable_null = nn.Parameter(torch.empty(1, n_tokens, token_dim))
        nn.init.normal_(self.learnable_null, std=token_dim**-0.5)
        self.out_norm = nn.LayerNorm(token_dim)

    @property
    def num_tokens(self) -> int:
        return self.num_params * self.num_tokens_per_param

    def forward(
        self,
        params: torch.Tensor,
        *,
        train: bool = False,
        force_null: bool | torch.Tensor = False,
    ) -> torch.Tensor:
        """
        Args:
            params: (B, num_params) normalized floats in roughly [-1, 1]
        Returns:
            tokens: (B, num_params * num_tokens_per_param, cross_attention_dim)
        """
        if params.ndim != 2 or params.shape[1] != self.num_params:
            raise ValueError(
                f"Expected params shape (B, {self.num_params}), got {tuple(params.shape)}"
            )
        bsz = params.shape[0]
        dtype = self.value_mlp[0].weight.dtype
        values = params.unsqueeze(-1).to(dtype=dtype)  # (B, N, 1)
        tokens = self.value_mlp(values)  # (B, N, T*D)
        ids = torch.arange(self.num_params, device=params.device)
        tokens = tokens + self.param_id_embed(ids)[None].to(dtype=tokens.dtype)
        # (B, N, T, D) -> (B, N*T, D)
        tokens = tokens.view(bsz, self.num_tokens, self.cross_attention_dim)
        tokens = self.out_norm(tokens)

        null = self.learnable_null.to(dtype=tokens.dtype)
        if isinstance(force_null, torch.Tensor):
            if force_null.shape != (bsz,):
                raise ValueError(
                    f"force_null must have shape ({bsz},), got {tuple(force_null.shape)}"
                )
            tokens = torch.where(force_null[:, None, None], null, tokens)
            if train and self.drop_prob > 0:
                drop = (torch.rand(bsz, device=params.device) < self.drop_prob) & ~force_null
                tokens = torch.where(drop[:, None, None], null, tokens)
        else:
            if force_null:
                tokens = null.expand(bsz, -1, -1)
            elif train and self.drop_prob > 0:
                drop = torch.rand(bsz, device=params.device) < self.drop_prob
                tokens = torch.where(drop[:, None, None], null, tokens)
        return tokens
