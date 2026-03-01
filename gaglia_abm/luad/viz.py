"""Visualization utilities for Gaglia-grounded LUAD simulations."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Grid codes: 0=empty, 1=tumor, 2=CD8, 3=CD4, 4=Treg, 5=Macrophage
COLOR_MAP = {
    0: (0.0, 0.0, 0.0),       # empty
    1: (0.95, 0.4, 0.4),      # tumor
    2: (0.2, 0.6, 0.9),       # CD8
    3: (0.3, 0.8, 0.9),       # CD4
    4: (0.6, 0.2, 0.8),       # Treg
    5: (0.55, 0.55, 0.55),    # Macrophage
    6: (0.9, 0.9, 0.9),       # other
}

TRAJECTORY_PALETTE = {
    "tumor_count": "#F2625F",
    "cd8_count": "#3498DB",
    "cd4_count": "#4BC5DE",
    "treg_count": "#8E44AD",
    "macrophage_count": "#8F8F8F",
}


def plot_trajectories(timeseries: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    count_cols = [c for c in timeseries.columns if c.endswith("_count")]
    for column in count_cols:
        label = column.replace("_count", "")
        color = TRAJECTORY_PALETTE.get(column)
        ax.plot(timeseries.index, timeseries[column], label=label, linewidth=2.0, color=color)
    ax.set_xlabel("Tick", fontsize=12)
    ax.set_ylabel("Agent count", fontsize=12)
    ax.set_title("Population trajectories", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=False, fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    path = output_dir / "trajectories.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_distance_cdf(distance_records, output_dir: Path) -> Path:
    if not distance_records:
        return output_dir / "distance_cdf.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    for record in distance_records:
        ax.plot(record.hist_bins, record.hist_cdf, label=f"tick {record.tick}")
    ax.set_xlabel("CD8 -> tumor distance (cells)")
    ax.set_ylabel("CDF")
    ax.set_title("CD8 proximity to tumor")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    path = output_dir / "distance_cdf.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_interaction_heatmap(interaction_records, output_dir: Path) -> Path:
    if not interaction_records:
        return output_dir / "interaction_heatmap.png"
    latest = interaction_records[-1]
    pairs = sorted(latest.observed.keys())
    labels = sorted({label for pair in pairs for label in pair})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    mat = np.ones((len(labels), len(labels)))
    for (a, b), obs in latest.observed.items():
        expected = latest.expected.get((a, b), 1e-6)
        idx_a = label_to_idx[a]
        idx_b = label_to_idx[b]
        mat[idx_a, idx_b] = obs / (expected + 1e-6)
        mat[idx_b, idx_a] = mat[idx_a, idx_b]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="coolwarm", vmin=0.2, vmax=2.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Enrichment vs shuffled")
    ax.set_title(f"Interaction enrichment (tick {latest.tick})")
    path = output_dir / "interaction_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_movie(snapshots, output_dir: Path, fps: int = 10, scale: int = 6, legend: bool = True) -> Path:
    if not snapshots:
        return output_dir / "movie.gif"
    frames = []
    legend_panel = None
    for tick, grid in snapshots:
        rgb = grid_to_rgb(grid, scale=scale)
        if legend:
            if legend_panel is None:
                legend_panel = build_legend(rgb.shape[0], rgb.shape[1])
            rgb = np.concatenate([rgb, legend_panel], axis=1)
        frames.append((rgb * 255).astype(np.uint8))
    path = output_dir / "movie.gif"
    imageio.mimsave(path, frames, fps=fps)
    return path


def build_legend(height: int, frame_width: int) -> np.ndarray:
    entries = [
        ("Tumor", COLOR_MAP[1]),
        ("CD8 T", COLOR_MAP[2]),
        ("CD4 T", COLOR_MAP[3]),
        ("Treg", COLOR_MAP[4]),
        ("Macrophage", COLOR_MAP[5]),
    ]
    try:
        font_size = max(16, height // 10)
        font = ImageFont.truetype('Arial.ttf', size=font_size)
    except OSError:
        font = ImageFont.load_default()
    rows = len(entries)
    row_h = max(1, height // rows)
    box_h = int(0.55 * row_h)
    box_w = box_h
    text_metrics = []
    for label, _ in entries:
        bbox = font.getbbox(label)
        text_metrics.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))
    max_text_width = max((w for w, _ in text_metrics), default=0)
    width = max(400, box_w + 10 + 24 + max_text_width + 30)
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for idx, ((label, color), (text_w, text_h)) in enumerate(zip(entries, text_metrics)):
        y0 = idx * row_h
        y1 = height if idx == rows - 1 else (idx + 1) * row_h
        box_y0 = y0 + ((y1 - y0) - box_h) // 2
        box_y1 = box_y0 + box_h
        draw.rectangle([12, box_y0, 12 + box_w, box_y1], fill=tuple(int(c * 255) for c in color))
        text_x = 12 + box_w + 24
        text_y = y0 + ((y1 - y0) - text_h) // 2
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
    legend_arr = np.asarray(img, dtype=np.float32) / 255.0
    return legend_arr


def grid_to_rgb(grid: np.ndarray, scale: int = 1) -> np.ndarray:
    height, width = grid.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for code, color in COLOR_MAP.items():
        mask = grid == code
        rgb[mask] = color
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb
