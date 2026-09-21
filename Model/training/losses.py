"""
OceanEmbed loss functions:
masked MSE + spacing-aware smoothness + gradient-preservation + adaptive inversion penalty.
Deliberately excludes hard stability/monotonicity penalties and variance-normalized weighting
(see technical approach doc for why).
"""
import torch


def masked_mse(pred, target, mask):
    diff2 = (pred - target) ** 2 * mask
    return diff2.sum() / mask.sum().clamp(min=1)


def spacing_aware_smoothness(pred, depths, mask):
    """depths: 1D tensor of standard depths in meters, same order as pred's depth dim."""
    dz = (depths[1:] - depths[:-1]).view(1, -1, 1, 1).clamp(min=1e-3)
    grad = (pred[:, 1:] - pred[:, :-1]) / dz
    m = mask[:, 1:] if mask.dim() == pred.dim() else mask
    return ((grad ** 2) * m).mean()


def gradient_preservation_loss(pred, target, depths, mask):
    dz = (depths[1:] - depths[:-1]).view(1, -1, 1, 1).clamp(min=1e-3)
    grad_pred = (pred[:, 1:] - pred[:, :-1]) / dz
    grad_true = (target[:, 1:] - target[:, :-1]) / dz
    m = mask[:, 1:] if mask.dim() == pred.dim() else mask
    return (((grad_pred - grad_true) ** 2) * m).mean()


def adaptive_inversion_penalty(pred, inversion_freq_map, mask, climate_idx=None):
    """
    Penalizes candidate inversions (deeper warmer than shallower), weighted DOWN in
    locations where inversions are climatologically common (protects real BoB winter
    inversions) and weighted UP where they are climatologically rare (Arabian Sea).
    Optionally modulated by ENSO/IOD index (bounded via tanh).
    """
    diff = torch.relu(pred[:, 1:] - pred[:, :-1])
    modulation = 1.0
    if climate_idx is not None:
        modulation = 1.0 + 0.3 * torch.tanh(climate_idx)
    weight = (1 - inversion_freq_map * modulation).clamp(min=0.1)
    m = mask[:, 1:] if mask.dim() == pred.dim() else mask
    return ((diff ** 2) * weight.unsqueeze(0).unsqueeze(0) * m).mean()


def thermocline_band_weight(depths, band=(75, 150), weight=1.5):
    """Returns a per-depth multiplier tensor, 1.0 everywhere except the bounded thermocline band."""
    w = torch.ones_like(depths)
    band_mask = (depths >= band[0]) & (depths <= band[1])
    w[band_mask] = weight
    return w


def total_loss(pred, target, mask, depths, inversion_freq_map, climate_idx=None,
               lambda_smooth=0.1, lambda_grad=0.1, lambda_thermo=0.5,
               thermo_band=(75, 150), thermo_weight=1.5):
    recon = masked_mse(pred, target, mask)
    smooth = spacing_aware_smoothness(pred, depths, mask)
    grad_preserve = gradient_preservation_loss(pred, target, depths, mask)
    inv_penalty = adaptive_inversion_penalty(pred, inversion_freq_map, mask, climate_idx)

    total = recon + lambda_smooth * smooth + lambda_grad * grad_preserve + lambda_thermo * inv_penalty
    parts = {
        "recon": recon.item(), "smooth": smooth.item(),
        "grad_preserve": grad_preserve.item(), "inv_penalty": inv_penalty.item(),
        "total": total.item(),
    }
    return total, parts
