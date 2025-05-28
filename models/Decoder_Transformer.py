import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from .layers import (
    LinearBlock,
    PositionalEncoding,
    TransformerBlock)

class DecoderTransformer(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            hidden_dim,
            num_heads=8,
            out_classes=8,
            num_blocks=4,
            activation_type="gelu"):
        super().__init__()

        # Learnable Embedding and Positional Encoding.
        self.emb_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim)
        self.pos_layer = PositionalEncoding()

        # Decoder Blocks.
        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.decoder_blocks.append(
                TransformerBlock(
                    heads=num_heads,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    self_attn_is_causal=True,
                    activation_type=activation_type))

        # Classifier Block.
        self.classifier_block = nn.Sequential(
            LinearBlock(
                in_dim=embedding_dim,
                out_dim=hidden_dim,
                use_activation=True),
            LinearBlock(
                in_dim=hidden_dim,
                out_dim=embedding_dim,
                use_activation=False))

    def custom_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print(f"No Layer found: {name}, skipping")
                continue
            # Skip loading mismatched weights, in cases of weight changes.
            if (own_state[name].shape != param.data.shape):
                print(f"Skipped: {name}")
                continue
            if isinstance(param, torch.nn.parameter.Parameter):
                # Backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def forward(self, x):
        # Embedding Layer + Positional Encoding.
        x_emb = self.emb_layer(x)
        x_dec = self.pos_layer(x_emb)

        # Decoder Section.
        for decoder_block in self.decoder_blocks:
            x_dec = decoder_block(x_dec)  # (N,Seq,Dim)

        x_classifier = self.classifier_block(x_dec)
        return x_dec, x_classifier
