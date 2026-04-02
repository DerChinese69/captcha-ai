import torch
import torch.nn as nn

# -------------------------
# Patch Embedding
# -------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(64, 192), patch_size=(8, 16), in_chans=1, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size

        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # Conv = efficient patch extractor + linear projection
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, 1, 64, 192]
        x = self.proj(x)                    # [B, 128, 8, 12]
        x = x.flatten(2)                    # [B, 128, 96]
        x = x.transpose(1, 2)               # [B, 96, 128]
        return x


# -------------------------
# Transformer Encoder Block
# -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# -------------------------
# Small CAPTCHA ViT (Phase A)
# -------------------------
class SmallCaptchaViT(nn.Module):
    def __init__(
        self,
        img_size=(64, 192),
        patch_size=(8, 16),
        embed_dim=128,
        depth=4,
        num_heads=4,
        num_classes=10,
        label_length=5,
        dropout=0.1
    ):
        super().__init__()

        self.label_length = label_length
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer encoder
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, 4.0, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Head
        self.head = nn.Linear(embed_dim, label_length * num_classes)

    def forward(self, x):
        # [B, 1, 64, 192]
        x = self.patch_embed(x)                 # [B, 96, 128]
        x = x + self.pos_embed                 # add position

        x = self.blocks(x)                     # [B, 96, 128]
        x = self.norm(x)

        # Mean pooling
        x = x.mean(dim=1)                      # [B, 128]

        x = self.head(x)                       # [B, 255]
        x = x.view(-1, self.label_length, self.num_classes)  # [B, 5, 51]

        return x