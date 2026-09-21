"""
Stage 1: self-supervised pretraining of the shared encoder on unlabeled
surface-only data (2021-2023, non-overlapping with Stage 2's GLORYS window).
Multi-task: reconstruct the 7 surface variables + regress MLD climatology.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import PRETRAIN_YEARS, BATCH_SIZE, STAGE1_EPOCHS, LR_STAGE1, ENCODER_BASE_CH, CHECKPOINT_DIR
from model import TemporalEncoder, Encoder, HighResBranch, BottleneckAttention, GlobalContextBranch, ConvBlock, SubPixelUpsample
from dataset import SurfaceOnlyDataset
import os


class AutoencoderDecoder(nn.Module):
    """Throwaway decoder — only exists to give the encoder a training signal."""
    def __init__(self, base=32, out_ch=7):
        super().__init__()
        self.up2 = SubPixelUpsample(base * 4, base * 2)  # -> matches s2's channel count for concat
        self.d2 = ConvBlock(base * 4, base * 2)           # cat(up2_out[base*2], s2[base*2]) = base*4 in
        self.up1 = SubPixelUpsample(base * 2, base)        # -> matches s1's channel count for concat
        self.d1 = ConvBlock(base * 2, base)                # cat(up1_out[base], s1[base]) = base*2 in
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, s1, s2, s3):
        up2 = self.up2(s3)
        up2 = nn.functional.interpolate(up2, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d2(torch.cat([up2, s2], dim=1))

        up1 = self.up1(x)
        up1 = nn.functional.interpolate(up1, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.d1(torch.cat([up1, s1], dim=1))
        return self.out(x)


class MLDHead(nn.Module):
    """Small head regressing mixed-layer-depth climatology from the bottleneck embedding."""
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))

    def forward(self, s3):
        up = nn.functional.interpolate(s3, scale_factor=4, mode="bilinear", align_corners=False)
        return self.net(up).squeeze(1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SurfaceOnlyDataset(years=PRETRAIN_YEARS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    temporal = TemporalEncoder(in_ch=19, t_window=14, out_ch=ENCODER_BASE_CH).to(device)
    hires = HighResBranch(in_ch=ENCODER_BASE_CH, hidden=ENCODER_BASE_CH // 2).to(device)
    encoder = Encoder(in_ch=ENCODER_BASE_CH, base=ENCODER_BASE_CH).to(device)
    bn_attn = BottleneckAttention(dim=ENCODER_BASE_CH * 4).to(device)
    global_ctx = GlobalContextBranch().to(device)
    ae_decoder = AutoencoderDecoder(base=ENCODER_BASE_CH, out_ch=7).to(device)
    mld_head = MLDHead(in_ch=ENCODER_BASE_CH * 4).to(device)

    params = list(temporal.parameters()) + list(hires.parameters()) + list(encoder.parameters()) + \
             list(bn_attn.parameters()) + list(ae_decoder.parameters()) + list(mld_head.parameters())
    opt = torch.optim.Adam(params, lr=LR_STAGE1)
    mse = nn.MSELoss()

    for epoch in range(STAGE1_EPOCHS):
        running_loss = 0.0
        for batch in loader:
            x = batch["x"].to(device)                       # [B, 19, 14, H, W]
            target_surface = batch["target_surface"].to(device)  # [B, 7, H, W]

            feat = temporal(x)
            s1, s2, s3 = encoder(feat)
            s3_att = s3 + bn_attn(s3)
            s3_att = s3_att + global_ctx(s3_att)

            recon = ae_decoder(s1, s2, s3_att)
            loss_recon = mse(recon, target_surface)

            loss = loss_recon
            if "mld_target" in batch:
                mld_pred = mld_head(s3_att)
                loss_mld = mse(mld_pred, batch["mld_target"].to(device))
                loss = loss + 0.3 * loss_mld

            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()

        print(f"[Stage 1] Epoch {epoch+1}/{STAGE1_EPOCHS} - loss: {running_loss/len(loader):.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "temporal": temporal.state_dict(),
        "hires": hires.state_dict(),
        "encoder": encoder.state_dict(),
        "bn_attn": bn_attn.state_dict(),
    }, f"{CHECKPOINT_DIR}/stage1_pretrained.pt")
    print("Stage 1 pretraining complete, encoder saved.")


if __name__ == "__main__":
    main()
