# OceanEmbed — Model Architecture, In Depth

This document explains every component of the OceanEmbed neural network — what it does, why it exists, how it connects to the next piece, and the code behind it. Read top to bottom for the full flow from raw input to final temperature prediction.

---

## 0. The One-Sentence Version

**A CNN encoder compresses 14 days of 19-channel ocean surface data into a compact "embedding," and a decoder expands that embedding back out — not into 15 fixed depth channels, but into a continuous function that can be queried at any depth — producing full-depth ocean temperature as its final output.**

---

## 1. Overall Architecture Diagram (Text Form)

```
INPUT: [Batch, 19 channels, 14 days, Height, Width]
         │
         ▼
┌─────────────────────────┐
│  TEMPORAL FRONT-END      │   ← collapses 14 days into 1 feature map
│  (3D Conv / lagged-ch)   │
└───────────┬──────────────┘
            ▼
┌─────────────────────────┐
│       ENCODER            │   ← compresses spatial info, builds "embedding"
│  Conv blocks (downsample)│
│  + Bottleneck Attention   │
│  + Global Context Pooling │
└───────────┬──────────────┘
            ▼
    [compact embedding + skip connections s1, s2, s3]
            │
            ▼
┌─────────────────────────┐
│       DECODER             │   ← expands embedding back to full resolution
│  Attention-gated skips    │
│  Sub-pixel upsampling     │
│  Parallel HR detail branch│
└───────────┬──────────────┘
            ▼
    [per-pixel feature map, still at full spatial resolution]
            │
            ▼
┌─────────────────────────┐
│  IMPLICIT DEPTH DECODER   │   ← query this feature map AT a depth value
│  (Fourier-encoded depth   │      e.g. "give me temp at 125m"
│   + small MLP)            │
└───────────┬──────────────┘
            ▼
    ΔT prediction (anomaly from climatology) at queried depth
            │
            ▼
┌─────────────────────────┐
│  CLIMATOLOGY ANCHOR HEAD  │   ← adds back the seasonal average
└───────────┬──────────────┘
            ▼
OUTPUT: Absolute temperature at that depth, for every pixel
(repeat the last two steps for all 15 standard depths)
```

---

## 2. Temporal Front-End

**Purpose:** Collapse the 14-day input window into a single feature map before the main encoder sees it, so the network has "memory" of recent ocean forcing (wind buildup, slow warming trends) rather than judging purely on a single day's snapshot.

**How it works:** A 3D convolution slides across the time dimension as well as the two spatial dimensions, learning which patterns across the last 14 days matter, and compresses the time axis down to 1.

```python
class TemporalEncoder(nn.Module):
    def __init__(self, in_ch=19, t_window=14):
        super().__init__()
        self.temporal_conv = nn.Conv3d(
            in_ch, 32,
            kernel_size=(t_window, 3, 3),
            padding=(0, 1, 1)
        )
    def forward(self, x):  # x: [B, 19, 14, H, W]
        x = self.temporal_conv(x).squeeze(2)  # → [B, 32, H, W]
        return x
```

**Fallback (if compute-constrained):** Instead of true 3D convolution, use lagged snapshots as extra channels (today, 7 days ago, 14 days ago) stacked into the regular 2D input — cheaper, less expressive, listed as the first thing to cut under time pressure.

---

## 3. Encoder — Building the "Satellite Embedding"

**Purpose:** This is the part that directly answers the problem statement's requirement for a "compact satellite embedding." It progressively compresses the spatial map, extracting increasingly abstract patterns (from "this pixel is warm" to "this whole region has an eddy").

### 3.1 Basic Conv Blocks
```python
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU())
    def forward(self, x): return self.net(x)
```
Each block does two convolutions (each one looks at a small neighborhood of pixels and combines them into a new value), followed by normalization (GroupNorm, not BatchNorm — chosen specifically because BatchNorm's statistics get corrupted by the many zero-filled land pixels in ocean data) and a ReLU activation (keeps only positive signals, discards negative "noise").

### 3.2 Downsampling
Between blocks, a pooling operation shrinks the map's height and width by half each time, while increasing the number of feature channels — trading spatial detail for more abstract pattern information, layer by layer.

```python
class Encoder(nn.Module):
    def __init__(self, in_ch=32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        s1 = self.e1(x)                 # full resolution, 32 channels
        s2 = self.e2(self.pool(s1))     # half resolution, 64 channels
        s3 = self.e3(self.pool(s2))     # quarter resolution, 128 channels
        return s1, s2, s3   # s1, s2 saved as "skip connections" for later
```
`s1` and `s2` are kept aside — they still contain fine spatial detail that gets lost by the time we reach `s3`. They'll be reintroduced in the decoder so we don't throw that detail away permanently.

### 3.3 Bottleneck Self-Attention
**Purpose:** Fixes the "fixed receptive field" problem — plain convolutions only ever look at nearby pixels, so they can miss basin-scale patterns like a large eddy or a Rossby wave that spans a big chunk of the map. At the most-compressed stage (`s3`), the map is small enough that full self-attention (every pixel checking in with every other pixel) becomes computationally cheap.

```python
class BottleneckAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)      # turn the 2D map into a list of tokens
        out, _ = self.attn(tokens, tokens, tokens)   # each token "looks at" every other token
        return out.permute(0, 2, 1).reshape(B, C, H, W)
```

### 3.4 Global Context Pooling
**Purpose:** A second, complementary way to give the model basin-wide awareness — instead of pixel-to-pixel attention, this simply averages the entire map into one summary vector ("what's the whole basin doing today, on average") and broadcasts that back to every pixel, so local decisions can be informed by global state.

```python
class GlobalContextBranch(nn.Module):
    def forward(self, x):  # x: [B, C, H, W]
        global_vec = x.mean(dim=[2, 3], keepdim=True)   # average over space
        return global_vec.expand_as(x)                   # broadcast back to every pixel
```

---

## 4. Decoder — Reconstructing Detail

**Purpose:** Takes the compressed, abstract embedding and expands it back out to full spatial resolution, while reintroducing the fine detail saved earlier (`s1`, `s2`) so the output isn't blurry.

### 4.1 Attention Gates
**Purpose:** When reintroducing `s1`/`s2` detail, an attention gate decides *how much* of that detail to actually use at each location — rather than blindly copying it all back in, which can reintroduce noise.

```python
class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.wg = nn.Conv2d(f_g, f_int, 1)
        self.wx = nn.Conv2d(f_l, f_int, 1)
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.Sigmoid())
    def forward(self, g, x):
        # g = signal coming from the decoder (what we're building)
        # x = the saved skip-connection detail (s1 or s2)
        att = self.psi(torch.relu(self.wg(g) + self.wx(x)))  # a 0-to-1 "how relevant" map
        return x * att   # scale the skip detail by its relevance
```

### 4.2 Sub-pixel (PixelShuffle) Upsampling
**Purpose:** Standard "transpose convolution" upsampling (the most common way to grow a small feature map back to full size) is known to create blur and checkerboard-pattern artifacts. Sub-pixel upsampling avoids this by rearranging channels into spatial positions instead of learning to "spread out" values — a cleaner mathematical operation with fewer artifacts.

```python
class SubPixelUpsample(nn.Module):
    def __init__(self, channels, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
    def forward(self, x):
        return self.shuffle(self.conv(x))
```

### 4.3 Parallel High-Resolution Detail Branch
**Purpose:** A separate, small path that runs alongside the main encoder-decoder but **never gets downsampled** — it processes the original full-resolution input the whole time. This gives the network one path where fine spatial detail (like a sharp thermocline gradient) is never lost to compression in the first place, then fuses this detail back in at the very end.

```python
class HighResBranch(nn.Module):
    def __init__(self, in_ch, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU())
    def forward(self, x):
        return self.net(x)   # stays at full resolution throughout
```

### 4.4 Full Decoder, Assembled
```python
class AttentionUNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up2 = SubPixelUpsample(128)
        self.att2 = AttentionGate(64, 64, 32)
        self.d2 = ConvBlock(128, 64)
        self.up1 = SubPixelUpsample(64)
        self.att1 = AttentionGate(32, 32, 16)
        self.d1 = ConvBlock(64, 32)

    def forward(self, s1, s2, s3, hires_features):
        g2 = self.up2(s3)
        a2 = self.att2(g2, s2)
        x = self.d2(torch.cat([g2, a2], dim=1))

        g1 = self.up1(x)
        a1 = self.att1(g1, s1)
        x = self.d1(torch.cat([g1, a1], dim=1))

        x = x + hires_features   # fuse in the never-downsampled detail path
        return x   # final feature map, full resolution, ready for depth querying
```

---

## 5. Implicit Depth Decoder — The Key Innovation

**Purpose:** Solves the "depth as unordered channels" problem. Instead of outputting 15 fixed, disconnected channels, this module treats depth as a continuous *input* — you feed it a specific depth value, and it predicts temperature at exactly that depth. This naturally respects the real, non-uniform spacing between standard depths (5m gaps near the surface vs. 300m gaps at depth) because it works with the actual numbers, not arbitrary channel positions.

### 5.1 Why Depth Can't Be Fed In Raw
Feeding the number "1000" (meters) alongside the number "5" (meters) directly into a neural network creates a scale imbalance — 1000 is 200 times larger than 5, which distorts training. The fix, borrowed from how Transformers encode word position, is **Fourier encoding**: convert the depth into a set of sine/cosine waves at different frequencies, putting every depth value on a comparable numerical scale.

```python
class ImplicitDepthDecoder(nn.Module):
    def __init__(self, feat_dim=32, hidden=64, n_freqs=8):
        super().__init__()
        self.n_freqs = n_freqs
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 2 * n_freqs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def fourier_encode(self, depth_value):
        freqs = 2.0 ** torch.arange(self.n_freqs)     # e.g. 1, 2, 4, 8, 16...
        args = depth_value * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, feat_map, depth_value):
        B, C, H, W = feat_map.shape
        d_enc = self.fourier_encode(depth_value)
        d_enc = d_enc.view(1, 1, 1, -1).expand(B, H, W, -1)
        x = torch.cat([feat_map.permute(0, 2, 3, 1), d_enc], dim=-1)
        return self.mlp(x).squeeze(-1)   # [B, H, W] — temperature AT this depth
```

### 5.2 How It's Used at Inference
To get the full 15-depth profile, you call this same small network 15 times (once per standard depth), each time passing in a different `depth_value` — or better, batch all 15 depth queries together for one efficient vectorized call rather than 15 sequential ones.

---

## 6. Climatology Anchor Head — Final Output Step

**Purpose:** The network internally learns to predict **ΔT** (an anomaly — how different today's temperature is from the normal seasonal average at that depth/location), not the raw absolute temperature. This makes deep-ocean predictions more stable, since the model can default toward "probably close to normal" instead of inventing an unsupported guess where surface data has little information. But the problem statement asks for **absolute temperature**, so this head adds the climatology back in automatically, inside the model itself.

```python
class ClimatologyAnchoredHead(nn.Module):
    def __init__(self, climatology_field):
        super().__init__()
        # climatology_field: [15 depths, H, W, 12 months] — precomputed once from GLORYS training data
        self.register_buffer("clim", climatology_field)

    def forward(self, delta_t_pred, month_idx):
        clim_this_month = self.clim[..., month_idx]
        return delta_t_pred + clim_this_month   # → absolute temperature
```

**Rolling version (production-ready refinement):** instead of a permanently fixed climatology, this buffer is periodically updated with an exponentially-weighted rolling average, so the model doesn't silently go stale years after training:

```python
class RollingClimatology:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.clim = None
    def update(self, new_year_monthly_means):
        self.clim = new_year_monthly_means if self.clim is None else \
            (1 - self.alpha) * self.clim + self.alpha * new_year_monthly_means
    def get(self, month_idx):
        return self.clim[..., month_idx]
```

---

## 7. Full Forward Pass, End to End

```python
class OceanEmbedModel(nn.Module):
    def __init__(self, climatology_field):
        super().__init__()
        self.temporal = TemporalEncoder()
        self.encoder = Encoder()
        self.bottleneck_attn = BottleneckAttention(dim=128)
        self.global_ctx = GlobalContextBranch()
        self.hires_branch = HighResBranch(in_ch=32)
        self.decoder = AttentionUNetDecoder()
        self.depth_decoder = ImplicitDepthDecoder()
        self.clim_head = ClimatologyAnchoredHead(climatology_field)

    def forward(self, x_seq, depth_values, month_idx):
        # x_seq: [B, 19, 14, H, W]  — 14-day windowed input
        x = self.temporal(x_seq)               # → [B, 32, H, W]
        hires = self.hires_branch(x)           # never-downsampled detail path

        s1, s2, s3 = self.encoder(x)
        s3 = self.bottleneck_attn(s3)
        s3 = s3 + self.global_ctx(s3)          # inject basin-wide summary

        feat_map = self.decoder(s1, s2, s3, hires)

        outputs = []
        for depth in depth_values:              # e.g. all 15 standard depths
            delta_t = self.depth_decoder(feat_map, depth)
            absolute_t = self.clim_head(delta_t, month_idx)
            outputs.append(absolute_t)

        return torch.stack(outputs, dim=1)      # [B, 15, H, W] — final temperature field
```

---

## 8. Why Each Component Exists — Summary Table

| Component | Problem It Solves |
|---|---|
| Temporal front-end (3D conv, 14-day window) | Snapshot blindness — today's subsurface state depends on weeks of prior forcing |
| GroupNorm (not BatchNorm) | Land-cell zero-fill corrupting batch statistics |
| Bottleneck self-attention | Fixed receptive field missing basin-scale eddies/Rossby waves |
| Global context pooling | Same as above, complementary mechanism |
| Attention gates in decoder | Blindly copying skip-connection detail reintroduces noise |
| Sub-pixel upsampling | Transpose-conv blur/checkerboard artifacts |
| Parallel high-res branch | Thermocline sharpness lost through compression bottleneck |
| Implicit Fourier depth decoder | Depth treated as unordered/unrelated channels |
| Climatology anchor head | Deep-level noise chasing; also satisfies "output absolute T" requirement |
| Rolling climatology | Static buffer going stale over multi-year deployment |

---

## 9. What Feeds Into This Architecture (Recap)

Each of the 14 days in `x_seq` is a 19-channel stack: 7 real surface variables (SST, SSS, SSH, U/V current, U/V wind), 3 CoordConv channels (lat, lon, bathymetry), 2 seasonal channels (day-of-year sine/cosine), and 7 validity masks (one per real variable) — see the companion data-pipeline document for the full breakdown of these input channels.
