import math

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Sinusoidal Position embeddings.
"""
class PositionalEncoding(nn.Module):
    def forward(self, x):
        N, Seq, D = x.shape
        pos_number = torch.arange(1, Seq + 1)

        half_dim = D // 2
        pos_emb = math.log(10_000) / (half_dim - 1)
        pos_emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) * -pos_emb)

        pos_emb = pos_number[:, None] * pos_emb[None, :]
        pos_emb = torch.cat((pos_emb.sin(), pos_emb.cos()), dim=1)
        pos_emb = pos_emb.unsqueeze(0).to(x.device)  # (1, Seq, D)

        x = x + pos_emb

        return x


"""
Gaussian Error Linear Unit (GELU).
Shown to improve performance in Transformer architecture.
https://arxiv.org/abs/1606.08415
"""
class GELU(nn.Module):
    def forward(self, x):
        x_gelu = (0.5 * x) * (
            1 + torch.tanh(
                ((2 / math.pi)**0.5) * (x + 0.044715 * torch.pow(x, 3))))
        return x_gelu


"""
List of activation to be used.
"""
def get_activation(activation_type):
    activations_dict = nn.ModuleDict([
        ['gelu', GELU()],
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU()],
        ['sigmoid', nn.Sigmoid()],
        ['tanh', nn.Tanh()],
    ])
    return activations_dict[activation_type]


"""
Linear layers.
"""
class LinearBlock(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            use_activation=True,
            activation_type="gelu"):
        super().__init__()

        linear_list = [
            nn.Linear(in_dim, out_dim)
        ]

        if use_activation:
            linear_list.append(
                get_activation(activation_type=activation_type))

        self.linear_layer = nn.Sequential(*linear_list)

    def forward(self, x):
        x = self.linear_layer(x)
        return x


"""
Residual Linear Layers.
"""
class ResidualLinearBlock(nn.Module):
    def __init__(
            self,
            dim=512,
            activation_type="gelu"):
        super().__init__()

        self.linear = LinearBlock(
            in_dim=dim,
            out_dim=dim,
            use_activation=True,
            activation_type=activation_type)
        self.activation = get_activation(activation_type=activation_type)

    def forward(self, x, x_skip):
        x = self.linear(x)

        # Skip connection
        x = x + x_skip

        x = self.activation(x)
        return x


"""
Self-Attention Block.
"""
class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            heads=8,
            embedding_dim=512,
            hidden_dim=2_048,
            is_causal=True,
            activation_type="gelu"):
        super().__init__()

        self.heads = heads
        self.is_causal = is_causal

        self.q_block = nn.Sequential(
            LinearBlock(
                in_dim=embedding_dim,
                out_dim=hidden_dim,
                use_activation=True,
                activation_type=activation_type),
            LinearBlock(
                in_dim=hidden_dim,
                out_dim=embedding_dim,
                use_activation=False))
        self.k_block = nn.Sequential(
            LinearBlock(
                in_dim=embedding_dim,
                out_dim=hidden_dim,
                use_activation=True,
                activation_type=activation_type),
            LinearBlock(
                in_dim=hidden_dim,
                out_dim=embedding_dim,
                use_activation=False))
        self.v_block = nn.Sequential(
            LinearBlock(
                in_dim=embedding_dim,
                out_dim=hidden_dim,
                use_activation=True,
                activation_type=activation_type),
            LinearBlock(
                in_dim=hidden_dim,
                out_dim=embedding_dim,
                use_activation=True,
                activation_type=activation_type))

    def forward(self, x):
        q = self.q_block(x)  # (N, Seq_q, D)
        k = self.k_block(x)  # (N, Seq_k, D)
        v = self.v_block(x)  # (N, Seq_v, D)

        N, Seq, D = q.shape
        D_split = D // self.heads

        # Logically split into heads.
        # (N, H, Seq, D_split)
        q_head_split = q.reshape(
            N, Seq, self.heads, D_split).permute(0, 2, 1, 3)
        k_head_split = k.reshape(
            N, Seq, self.heads, D_split).permute(0, 2, 1, 3)
        v_head_split = v.reshape(
            N, Seq, self.heads, D_split).permute(0, 2, 1, 3)

        # (N, H, Seq_q, D_split),(N, H, Seq_k, D_split) => (N, H, Seq_q, Seq_k)
        qk_T = torch.einsum("nhqd,nhkd->nhqk", q_head_split, k_head_split)
        qk_T_normalized = qk_T / (D_split**0.5)  # (N, H, Seq_q, Seq_k)

        if self.is_causal:
            _, Seq, _ = x.shape
            # (1, 1, Seq, Seq)
            mask = torch.ones((1,1,Seq,Seq), device=q.device)
            mask = torch.triu(mask, diagonal=1)
            mask_bool = mask.bool()

            # (N, H, Seq_q, Seq_k)
            qk_T_normalized.masked_fill_(mask_bool, -torch.inf)

        qk_T_softmax = F.softmax(qk_T_normalized, dim=3)  # (N, H, Seq_q, Seq_k)

        # (N, H, Seq, D_split)
        attention_out = torch.einsum("nhqk,nhkd->nhqd", qk_T_softmax, v_head_split)

        # Merge multi-head computations.
        N, Head, Seq, Dsplit = attention_out.shape
        attention_out = attention_out.permute(0, 2, 1, 3).reshape(N, Seq, Head*Dsplit)  # (N, Seq, D)
        return attention_out


"""
Transformer Block.
"""
class TransformerBlock(nn.Module):
    def __init__(
            self,
            heads=8,
            hidden_dim=512,
            embedding_dim=512,
            self_attn_is_causal=False,
            activation_type="gelu"):
        super().__init__()

        # Self Attention Block.
        self.self_attn_block = SelfAttentionBlock(
            heads=heads,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            is_causal=self_attn_is_causal,
            activation_type=activation_type)
        self.self_attn_ffn_res = ResidualLinearBlock(
            dim=embedding_dim,
            activation_type=activation_type)
        self.self_attn_norm = nn.LayerNorm(embedding_dim)

        # FeedForward Layer.
        self.feed_forward = nn.Sequential(
            LinearBlock(
                in_dim=embedding_dim,
                out_dim=hidden_dim,
                use_activation=True,
                activation_type=activation_type),
            LinearBlock(
                in_dim=hidden_dim,
                out_dim=embedding_dim,
                use_activation=True,
                activation_type=activation_type),
            nn.LayerNorm(embedding_dim))
        self.feed_forward_res = ResidualLinearBlock(
            dim=embedding_dim,
            activation_type=activation_type)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x_self_attn = self.self_attn_block(x)
        x_self_attn_res = self.self_attn_ffn_res(x_self_attn, x)
        x_self_attn_norm = self.self_attn_norm(x_self_attn_res)

        x_ff = self.feed_forward(x_self_attn_norm)
        x_ff_res = self.feed_forward_res(x_ff, x_self_attn_norm)
        x_ff_norm = self.feed_forward_norm(x_ff_res)

        return x_ff_norm
