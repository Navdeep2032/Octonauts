import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

SKILL = {
    "depth_m": DEPTHS,
    "ensemble_skill": [0.0751, 0.0774, 0.0795, 0.0781, 0.0911, 0.1658, 0.2381, 0.2583, 0.2526, 0.2368, 0.1857, 0.0581, 0.0228, 0.0238, 0.0194],
    "clim_rmse": [0.7200, 0.7061, 0.6968, 0.7456, 0.8441, 1.2135, 1.7755, 2.0065, 1.7943, 1.4408, 0.9090, 0.5461, 0.3484, 0.3382, 0.3449],
}

ARGO = {
    "depth_m": DEPTHS,
    "single_rmse": [0.5060, 0.6832, 0.9109, 0.9314, 1.0722, 1.3491, 1.4669, 1.5061, 1.4417, 1.3141, 1.0133, 0.9881, 1.2116, 1.2353, 1.1379],
    "ensemble_rmse": [0.4805, 0.6748, 0.9108, 0.9391, 1.0713, 1.3533, 1.4558, 1.5028, 1.4216, 1.2981, 1.0066, 0.9885, 1.2125, 1.2334, 1.1337],
    "glorys_rmse": [0.2479, 0.3577, 0.5354, 0.5358, 0.6054, 0.8219, 0.9156, 1.0085, 0.9529, 0.8289, 0.5755, 0.3774, 0.2949, 0.2932, 0.2417],
}

SUMMARY = {
    "metric": [
        "Official ensemble GLORYS RMSE (C)",
        "Climatology RMSE (C)",
        "Overall GLORYS skill score",
        "Official ensemble ARGO RMSE (C)",
        "Legacy single-model ARGO RMSE (C)",
        "GLORYS vs ARGO RMSE (C)",
        "Ensemble gain on ARGO (C)",
    ],
    "value": [0.8746, 1.0926, 0.1995, 1.1859, 1.1906, 0.6476, 0.0047],
}


def save_table_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_skill_plot(output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(DEPTHS, SKILL["ensemble_skill"], color="#2E7D32")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("OceanEmbed ensemble skill score vs climatology by depth")
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Skill score")
    ax.set_xticks(DEPTHS)
    ax.set_xticklabels([str(d) for d in DEPTHS], rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_argo_plot(output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(DEPTHS))
    ax.plot(x, ARGO["single_rmse"], marker="o", label="Single model RMSE", linewidth=2)
    ax.plot(x, ARGO["ensemble_rmse"], marker="s", label="Ensemble RMSE", linewidth=2)
    ax.plot(x, ARGO["glorys_rmse"], marker="^", label="GLORYS RMSE", linewidth=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(d) for d in DEPTHS], rotation=45)
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("ARGO validation RMSE by depth")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_main_model_rmse_plot(output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    model_rmse = [0.6659, 0.6514, 0.6414, 0.6874, 0.7672, 1.0123, 1.3528, 1.4882, 1.3411, 1.0996, 0.7402, 0.5144, 0.3405, 0.3302, 0.3382]
    ax.plot(DEPTHS, model_rmse, marker="o", color="#1f77b4", linewidth=2.5, markersize=5)
    ax.set_title("OceanEmbed main-model RMSE by depth")
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_xticks(DEPTHS)
    ax.set_xticklabels([str(d) for d in DEPTHS], rotation=45)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    out_dir = Path("D:/Octonauts/plots")
    out_dir.mkdir(exist_ok=True)

    skill_rows = [{"depth_m": d, "ensemble_skill": s, "climatology_rmse": c} for d, s, c in zip(DEPTHS, SKILL["ensemble_skill"], SKILL["clim_rmse"]) ]
    argo_rows = [{
        "depth_m": d,
        "single_rmse": s,
        "ensemble_rmse": e,
        "glorys_rmse": g,
    } for d, s, e, g in zip(DEPTHS, ARGO["single_rmse"], ARGO["ensemble_rmse"], ARGO["glorys_rmse"])]

    save_table_csv(out_dir / "depth_skill_table.csv", skill_rows, ["depth_m", "ensemble_skill", "climatology_rmse"])
    save_table_csv(out_dir / "argo_rmse_table.csv", argo_rows, ["depth_m", "single_rmse", "ensemble_rmse", "glorys_rmse"])

    summary_rows = [{"metric": m, "value": v} for m, v in zip(SUMMARY["metric"], SUMMARY["value"])]
    save_table_csv(out_dir / "overall_summary_table.csv", summary_rows, ["metric", "value"])

    make_skill_plot(out_dir / "depth_skill_score.png")
    make_argo_plot(out_dir / "argo_rmse_by_depth.png")
    make_main_model_rmse_plot(out_dir / "main_model_rmse_by_depth.png")

    markdown = """# OceanEmbed final results summary

## Overall metrics

| Metric | Value |
| --- | ---: |
| Official ensemble GLORYS RMSE (C) | 0.8746 |
| Climatology RMSE (C) | 1.0926 |
| Overall GLORYS skill score | 0.1995 |
| Official ensemble ARGO RMSE (C) | 1.1859 |
| Legacy single-model ARGO RMSE (C) | 1.1906 |
| GLORYS vs ARGO RMSE (C) | 0.6476 |
| Ensemble gain on ARGO (C) | 0.0047 |

## Per-depth skill score

| Depth (m) | Skill |
| ---: | ---: |
| 0 | 0.0751 |
| 5 | 0.0774 |
| 10 | 0.0795 |
| 20 | 0.0781 |
| 30 | 0.0911 |
| 50 | 0.1658 |
| 75 | 0.2381 |
| 100 | 0.2583 |
| 125 | 0.2526 |
| 150 | 0.2368 |
| 200 | 0.1857 |
| 300 | 0.0581 |
| 500 | 0.0228 |
| 700 | 0.0238 |
| 1000 | 0.0194 |

## ARGO RMSE by depth

| Depth (m) | Single RMSE | Ensemble RMSE | GLORYS RMSE |
| ---: | ---: | ---: | ---: |
| 0 | 0.5060 | 0.4805 | 0.2479 |
| 5 | 0.6832 | 0.6748 | 0.3577 |
| 10 | 0.9109 | 0.9108 | 0.5354 |
| 20 | 0.9314 | 0.9391 | 0.5358 |
| 30 | 1.0722 | 1.0713 | 0.6054 |
| 50 | 1.3491 | 1.3533 | 0.8219 |
| 75 | 1.4669 | 1.4558 | 0.9156 |
| 100 | 1.5061 | 1.5028 | 1.0085 |
| 125 | 1.4417 | 1.4216 | 0.9529 |
| 150 | 1.3141 | 1.2981 | 0.8289 |
| 200 | 1.0133 | 1.0066 | 0.5755 |
| 300 | 0.9881 | 0.9885 | 0.3774 |
| 500 | 1.2116 | 1.2125 | 0.2949 |
| 700 | 1.2353 | 1.2334 | 0.2932 |
| 1000 | 1.1379 | 1.1337 | 0.2417 |

## Figures

- `plots/depth_skill_score.png`
- `plots/argo_rmse_by_depth.png`
- `plots/main_model_rmse_by_depth.png`
"""
    (out_dir / "final_summary.md").write_text(markdown, encoding="utf-8")

    print(f"Saved plots and tables to: {out_dir}")


if __name__ == "__main__":
    main()
