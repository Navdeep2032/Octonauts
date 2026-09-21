"""
OceanEmbed model architecture:
Temporal front-end -> Encoder (+ bottleneck attention + global context)
-> Attention U-Net-style decoder (sub-pixel upsampling + parallel hi-res branch)
-> Implicit Fourier depth-conditioned decoder -> Climatology anchor head.
"""
import torch
import torch.nn as nn


# ---------------- Temporal front-end ----------------
class TemporalEncoder(nn.Module):
    def __init__(self, in_ch=19, t_window=14, out_ch=32):
        super().__init__()
        self.temporal_conv = nn.Conv3d(in_ch, out_ch, kernel_size=(t_window, 3, 3), padding=(0, 1, 1))

    def forward(self, x):  # x: [B, 19, 14, H, W]
        return self.temporal_conv(x).squeeze(2)  # -> [B, out_ch, H, W]


# ---------------- Basic building blocks ----------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, groups=8):
        super().__init__()
        groups = min(groups, cout)
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BottleneckAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)
        out, _ = self.attn(tokens, tokens, tokens)
        return out.permute(0, 2, 1).reshape(B, C, H, W)


class GlobalContextBranch(nn.Module):
    def forward(self, x):
        return x.mean(dim=[2, 3], keepdim=True).expand_as(x)


class HighResBranch(nn.Module):
    def __init__(self, in_ch, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.out_ch = hidden

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.wg = nn.Conv2d(f_g, f_int, 1)
        self.wx = nn.Conv2d(f_l, f_int, 1)
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        att = self.psi(torch.relu(self.wg(g) + self.wx(x)))
        return x * att


class SubPixelUpsample(nn.Module):
    """Upsamples spatially by `scale` while mapping in_ch -> out_ch (out_ch typically
    smaller than in_ch, to match the skip connection it will be concatenated with)."""
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.out_ch = out_ch

    def forward(self, x):
        return self.shuffle(self.conv(x))


# ---------------- Encoder ----------------
class Encoder(nn.Module):
    def __init__(self, in_ch=32, base=32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.out_ch = base * 4

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        return s1, s2, s3


# ---------------- Decoder ----------------
class AttentionUNetDecoder(nn.Module):
    def __init__(self, base=32, hires_ch=16):
        super().__init__()
        # up2: s3 (base*4 ch, quarter-res) -> base*2 ch, half-res -- matches s2's channel count
        self.up2 = SubPixelUpsample(base * 4, base * 2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.d2 = ConvBlock(base * 4, base * 2)  # cat(g2[base*2], a2[base*2]) = base*4 in

        # up1: (base*2 ch, half-res) -> base ch, full-res -- matches s1's channel count
        self.up1 = SubPixelUpsample(base * 2, base)
        self.att1 = AttentionGate(base, base, base // 2)
        self.d1 = ConvBlock(base * 2, base)  # cat(g1[base], a1[base]) = base*2 in

        self.fuse_hires = nn.Conv2d(base + hires_ch, base, 1)
        self.out_ch = base

    def forward(self, s1, s2, s3, hires_features):
        g2 = self.up2(s3)
        g2 = nn.functional.interpolate(g2, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        a2 = self.att2(g2, s2)
        x = self.d2(torch.cat([g2, a2], dim=1))

        g1 = self.up1(x)
        g1 = nn.functional.interpolate(g1, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        a1 = self.att1(g1, s1)
        x = self.d1(torch.cat([g1, a1], dim=1))

        x = self.fuse_hires(torch.cat([x, hires_features], dim=1))
        return x


# ---------------- Implicit Fourier depth decoder ----------------
class ImplicitDepthDecoder(nn.Module):
    def __init__(self, feat_dim=32, hidden=64, n_freqs=8):
        super().__init__()
        self.n_freqs = n_freqs
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freqs).float())
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 2 * n_freqs, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def fourier_encode(self, depth_value):
        # depth_value: scalar tensor (e.g. 125.0)
        args = depth_value * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [2*n_freqs]

    def forward(self, feat_map, depth_value):
        B, C, H, W = feat_map.shape
        d_enc = self.fourier_encode(depth_value)                # [2*n_freqs]
        d_enc = d_enc.view(1, 1, 1, -1).expand(B, H, W, -1)      # broadcast to every pixel
        x = torch.cat([feat_map.permute(0, 2, 3, 1), d_enc], dim=-1)
        return self.mlp(x).squeeze(-1)  # [B, H, W]


# ---------------- Climatology anchor head ----------------
class ClimatologyAnchoredHead(nn.Module):
    def __init__(self, climatology_field):
        # climatology_field: [n_depths, H, W, 12] precomputed monthly climatology
        super().__init__()
        self.register_buffer("clim", climatology_field)

    def forward(self, delta_t_pred, depth_idx, month_idx):
        # self.clim: [n_depths, H, W, 12]. month_idx is a per-batch-item tensor [B],
        # so we must gather along the month axis per-sample, not do naive fancy indexing
        # (which would return [H, W, B] instead of [B, H, W] and silently broadcast wrong).
        clim_depth = self.clim[depth_idx].permute(2, 0, 1)   # [12, H, W]
        clim_this = clim_depth[month_idx]                     # [B, H, W], gathered per batch item
        return delta_t_pred + clim_this


class RollingClimatology:
    """Non-module helper for periodic climatology updates outside the forward pass."""
    def __init__(self, initial_clim, alpha=0.15):
        self.clim = initial_clim
        self.alpha = alpha

    def update(self, new_year_monthly_means):
        self.clim = (1 - self.alpha) * self.clim + self.alpha * new_year_monthly_means

    def get(self):
        return self.clim


# ---------------- Full model ----------------
class OceanEmbedModel(nn.Module):
    def __init__(self, climatology_field, in_ch=19, t_window=14, base=32,
                 n_freqs=8, depth_hidden=64, standard_depths=None):
        super().__init__()
        self.standard_depths = standard_depths or [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

        self.temporal = TemporalEncoder(in_ch=in_ch, t_window=t_window, out_ch=base)
        self.hires_branch = HighResBranch(in_ch=base, hidden=base // 2)
        self.encoder = Encoder(in_ch=base, base=base)
        self.bottleneck_attn = BottleneckAttention(dim=base * 4)
        self.global_ctx = GlobalContextBranch()
        self.decoder = AttentionUNetDecoder(base=base, hires_ch=base // 2)
        self.depth_decoder = ImplicitDepthDecoder(feat_dim=base, hidden=depth_hidden, n_freqs=n_freqs)
        self.clim_head = ClimatologyAnchoredHead(climatology_field)

    def forward(self, x_seq, month_idx, depths=None):
        """
        x_seq: [B, 19, 14, H, W]
        month_idx: [B] integer month index (0-11) for climatology lookup
        depths: list of depth values to query (defaults to all 15 standard depths)
        returns: [B, n_depths, H, W] absolute temperature
        """
        depths = depths or self.standard_depths
        x = self.temporal(x_seq)               # [B, base, H, W]
        hires = self.hires_branch(x)

        s1, s2, s3 = self.encoder(x)
        s3 = s3 + self.bottleneck_attn(s3)
        s3 = s3 + self.global_ctx(s3)

        feat_map = self.decoder(s1, s2, s3, hires)  # [B, base, H, W]

        outputs = []
        for depth_idx, depth_val in enumerate(depths):
            depth_tensor = torch.tensor(float(depth_val), device=x.device)
            delta_t = self.depth_decoder(feat_map, depth_tensor)         # [B, H, W]
            absolute_t = self.clim_head(delta_t, depth_idx, month_idx)    # [B, H, W]
            outputs.append(absolute_t)

        return torch.stack(outputs, dim=1)  # [B, n_depths, H, W]
