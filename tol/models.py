import math

import numpy as np
import torch
import torch.nn as nn


def l2_normalize(x, dim=1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-6)


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DirectionalPatchEncoder(nn.Module):
    def __init__(self, base_model, image_dim, text_dim, text_num=5, order="TNSWE"):
        super().__init__()
        self.base_model = base_model
        self.text_num = text_num
        self.order = order
        self.conv = ConvModule(image_dim, text_dim)
        self.last_tokens = None

        if hasattr(base_model, "visual"):
            self.backbone_type = "clip"
            base_model.visual.transformer.register_forward_hook(self._capture_tokens)
        elif hasattr(base_model, "vision_model"):
            self.backbone_type = "hf"
            base_model.vision_model.encoder.layers[-1].register_forward_hook(self._capture_tokens)
        else:
            raise ValueError(f"Unsupported backbone: {type(base_model).__name__}")

        if not hasattr(base_model, "logit_scale"):
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=False)

    def _capture_tokens(self, module, inputs, output):
        self.last_tokens = output[0] if isinstance(output, tuple) else output

    def _encode_image(self, images):
        if self.backbone_type == "clip":
            return self.base_model.encode_image(images)
        return self.base_model.get_image_features(pixel_values=images)

    def _encode_text(self, texts):
        if self.backbone_type == "clip":
            encode_text = self.base_model.encode_text
            if texts.dim() == 2:
                return encode_text(texts)
            return torch.cat([encode_text(texts[:, i]) for i in range(texts.shape[1])], dim=1)

        if texts.dim() == 2:
            return self.base_model.get_text_features(input_ids=texts)
        return torch.cat(
            [self.base_model.get_text_features(input_ids=texts[:, i]) for i in range(texts.shape[1])],
            dim=1,
        )

    def _tokens_to_grid(self, tokens, batch_size):
        if tokens is None:
            raise RuntimeError("No image tokens were captured from the backbone.")
        if tokens.dim() != 3:
            raise RuntimeError(f"Expected 3-D token tensor, got {tuple(tokens.shape)}")

        if tokens.shape[0] == batch_size:
            tokens = tokens
        elif tokens.shape[1] == batch_size:
            tokens = tokens.permute(1, 0, 2).contiguous()
        else:
            tokens = tokens.permute(1, 0, 2).contiguous()

        patch_tokens = tokens[:, 1:, :] if self.backbone_type == "clip" else tokens
        side = int(math.sqrt(patch_tokens.shape[1]))
        if side * side != patch_tokens.shape[1]:
            raise RuntimeError(f"Cannot reshape {patch_tokens.shape[1]} tokens into a square grid.")
        return patch_tokens.reshape(batch_size, side, side, patch_tokens.shape[2]).permute(0, 3, 1, 2).contiguous()

    def _order_tokens(self):
        if isinstance(self.order, str) and "," in self.order:
            return [token.strip() for token in self.order.split(",") if token.strip()]
        return list(self.order)

    def _split_regions(self, grid, center_ratio=0.33):
        tokens = self._order_tokens()
        batch, channels, height, width = grid.shape
        yy = torch.linspace(-1, 1, height, device=grid.device).view(1, 1, height, 1).expand(batch, 1, height, width)
        xx = torch.linspace(-1, 1, width, device=grid.device).view(1, 1, 1, width).expand(batch, 1, height, width)
        dx = xx
        dy = -yy
        center = (dx.abs() <= center_ratio) & (dy.abs() <= center_ratio)
        angle = (torch.atan2(dy, dx) * 180.0 / np.pi + 360.0) % 360.0

        if tokens == list("TNSWE"):
            masks = {
                "T": center,
                "N": (dy > 0) & (dy >= dx.abs()),
                "S": (dy < 0) & (-dy >= dx.abs()),
                "E": (dx > 0) & (dx > dy.abs()),
                "W": (dx < 0) & (-dx > dy.abs()),
            }
        else:
            def sector(center_deg, width_deg):
                delta = (angle - center_deg + 180.0) % 360.0 - 180.0
                return (~center) & (delta >= -width_deg / 2.0) & (delta < width_deg / 2.0)

            use_diagonals = any(token in tokens for token in ("NE", "SE", "SW", "NW"))
            cardinal_width = 45.0 if use_diagonals else 90.0
            masks = {
                "T": center,
                "E": sector(0.0, cardinal_width),
                "NE": sector(45.0, 45.0),
                "N": sector(90.0, cardinal_width),
                "NW": sector(135.0, 45.0),
                "W": sector(180.0, cardinal_width),
                "SW": sector(225.0, 45.0),
                "S": sector(270.0, cardinal_width),
                "SE": sector(315.0, 45.0),
            }

        unknown = [token for token in tokens if token not in masks]
        if unknown:
            raise ValueError(f"Unknown order token(s): {unknown}")

        def masked_mean(mask):
            mask = mask.float()
            denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            return (grid * mask).sum(dim=(2, 3), keepdim=True).div(denom).squeeze(-1).squeeze(-1)

        return torch.cat([masked_mean(masks[token]) for token in tokens], dim=1)

    def forward(self, images, texts, return_features=False):
        self.last_tokens = None
        _ = self._encode_image(images)
        patch_grid = self._tokens_to_grid(self.last_tokens, images.shape[0])
        patch_grid = patch_grid.to(dtype=self.conv.net[0].weight.dtype)
        image_map = self.conv(patch_grid)
        image_desc = l2_normalize(self._split_regions(image_map), dim=1)
        text_desc = l2_normalize(self._encode_text(texts).to(dtype=image_desc.dtype), dim=1)

        if return_features:
            return patch_grid, image_desc, text_desc

        logit_scale = self.base_model.logit_scale.exp() if hasattr(self.base_model, "logit_scale") else self.logit_scale.exp()
        logits_per_image = logit_scale * image_desc @ text_desc.t()
        return logits_per_image, logits_per_image.t()


class ImageTextRegistration(nn.Module):
    def __init__(self, ada_size=7, img_dim=768, text_dim=512, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.img_proj = nn.Conv2d(img_dim, embed_dim, kernel_size=1)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.pool = nn.AdaptiveAvgPool2d((ada_size, ada_size))
        self.norm_img_1 = nn.LayerNorm(embed_dim)
        self.norm_img_2 = nn.LayerNorm(embed_dim)
        self.norm_text = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ffn1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
        self.ffn2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))

    def forward(self, img_feat, text_feat):
        img_feat = self.pool(self.img_proj(img_feat))
        batch, channels, height, width = img_feat.shape
        x = img_feat.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        x_sa, _ = self.self_attn(self.norm_img_1(x), self.norm_img_1(x), self.norm_img_1(x), need_weights=False)
        x = x + self.drop(x_sa)
        x = x + self.ffn1(x)

        text_tokens = text_feat if text_feat.dim() == 3 else text_feat.unsqueeze(1)
        text_tokens = self.norm_text(self.text_proj(text_tokens))
        x_ca, _ = self.cross_attn(self.norm_img_2(x), text_tokens, text_tokens, need_weights=False)
        x = x + self.drop(x_ca)
        return x + self.ffn2(x)


class TOLLocalizationModel(nn.Module):
    def __init__(
        self,
        base_model,
        image_dim=768,
        text_dim=512,
        text_num=5,
        ada_size=7,
        embed_dim=512,
        num_heads=8,
        order="TNSWE",
    ):
        super().__init__()
        self.encoder = DirectionalPatchEncoder(base_model, image_dim=image_dim, text_dim=text_dim, text_num=text_num, order=order)
        self.registration = ImageTextRegistration(
            ada_size=ada_size,
            img_dim=image_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.offset_head = nn.Sequential(
            nn.Linear(embed_dim * ada_size * ada_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 2),
        )

    def forward_retrieval(self, images, texts):
        patch_grid, image_desc, text_desc = self.encoder(images, texts, return_features=True)
        logit_scale = self.encoder.base_model.logit_scale.exp() if hasattr(self.encoder.base_model, "logit_scale") else self.encoder.logit_scale.exp()
        logits_per_image = logit_scale * image_desc @ text_desc.t()
        return patch_grid, image_desc, text_desc, logits_per_image, logits_per_image.t()

    def forward_loc(self, image_features, text_features):
        image_features = image_features.to(dtype=self.registration.img_proj.weight.dtype)
        text_features = text_features.to(dtype=self.registration.text_proj.weight.dtype)
        batch, text_width = text_features.shape
        text_tokens = text_features.reshape(batch, self.encoder.text_num, text_width // self.encoder.text_num)
        fused = self.registration(image_features, text_tokens)
        offsets = self.offset_head(fused.flatten(1)).sigmoid() * 2.0 - 1.0
        return fused, offsets

    def forward(self, images, texts):
        image_features, image_desc, text_desc, logits_per_image, logits_per_text = self.forward_retrieval(images, texts)
        fused, offsets = self.forward_loc(image_features, text_desc)
        return fused, offsets, logits_per_image, logits_per_text
