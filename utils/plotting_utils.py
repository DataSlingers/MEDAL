import os
import re, ast
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Optional
import matplotlib.ticker as ticker
import numpy as np, pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt, seaborn as sns


# ----------------------------- helpers ---------------------------------

def find_band_checkpoints(base_dir: str | Path,
                          band_idx: int = 0,
                          pattern: str = "*_band_ckpts") -> List[Tuple[str, Path]]:
    """
    Returns [(variant_name, ckpt_path), ...] where variant_name is like 'tsne_10'
    from folders like 'tsne_10_band_ckpts/band0_stable.pt'.
    """
    base = Path(base_dir)
    items = []
    for d in base.glob(pattern):
        if not d.is_dir():
            continue
        variant = d.name.replace("_band_ckpts", "")  # e.g. 'tsne_10'
        ckpt = d / f"band{band_idx}_stable.pt"
        if ckpt.exists():
            items.append((variant, ckpt))
    return sorted(items, key=lambda x: x[0])


def variant_to_family_and_param(variant: str) -> Tuple[str, Optional[float]]:
    """
    Splits 'tsne_10' -> ('tsne', 10.0)
           'umap_15' -> ('umap', 15.0)
           'pca'     -> ('pca', None)
    Only used for labeling/sorting; robust to variants without a number.
    """
    m = re.match(r"([a-zA-Z]+)(?:[_\-])?(\d+(?:\.\d+)?)?$", variant)
    if not m:
        return variant, None
    fam = m.group(1)
    param = float(m.group(2)) if m.group(2) is not None else None
    return fam, param


def pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Given a list of (distill, recon), returns the non-dominated frontier
    when minimizing both. Sorted by distill ascending; we keep strictly
    improving recon as distill grows.
    """
    pts = [(d, r) for (d, r) in points if np.isfinite(d) and np.isfinite(r)]
    pts.sort(key=lambda x: x[0])
    front = []
    best_r = float("inf")
    for d, r in pts:
        if r < best_r:
            front.append((d, r))
            best_r = r
    return front


# ----------------------------- main API --------------------------------

def evaluate_band(
    base_dir: str | Path,
    band_idx: int,
    analysis_filepath: str | Path,
    tuning_param: str
):
    """
    Scans base_dir for '*_band_ckpts/band{band_idx}_stable.pt', evaluates each.
    Returns a list of dict rows.
    """
    out_rows = []
    analysis = pd.read_csv(f'/shared/share_mala/irchang/drd/compare_teachers/{analysis_filepath}')
    for variant, ckpt in find_band_checkpoints(base_dir, band_idx):
        fam, param = variant_to_family_and_param(variant)
        cond = (analysis["config/teacher_config/teacher"] == fam) & (analysis[f"config/teacher_config/{tuning_param}"] == param)
        d = analysis.loc[cond, "distill_loss"].values
        r = analysis.loc[cond, "recon_loss"].values
        
        out_rows.append({
            "variant": variant,
            "family": fam,
            "param": param,
            "band": band_idx,
            "distill": d,
            "recon": r,
            "ckpt_path": str(ckpt),
        })
    return out_rows


def plot_recon_vs_distill(results: List[dict], title: str = "Recon vs Distill (band-matched)"):
    """
    Primary figure: Recon (y) vs Distill (x, log scale).
    One line per teacher family connecting that family's Pareto front.
    """
    plt.figure(figsize=(8, 5))
    fams = sorted(set(r["family"] for r in results))
    for fam in fams:
        pts = [(r["distill"], r["recon"]) for r in results if r["family"] == fam]
        front = pareto_front(pts)
        # scatter all points for the family
        xs_all = [p[0] for p in pts if np.isfinite(p[0])]
        ys_all = [p[1] for p in pts if np.isfinite(p[1])]
        plt.scatter(xs_all, ys_all, label=fam, alpha=0.6)
        # connect the Pareto front for that family
        if len(front) >= 2:
            xs, ys = zip(*front)
            plt.plot(xs, ys, linewidth=2)  # no specific colors per your constraints
    plt.xscale("log")
    plt.xlabel("Distill loss (log scale)")
    plt.ylabel("Reconstruction loss")
    plt.title(title)
    plt.legend(title="Teacher family")
    plt.tight_layout()


def plot_recon_bar_at_band(results: List[dict], title: str = "Recon at matched band"):
    """
    Secondary figure: Bar chart of recon at this band grouped by family.
    With 1 seed, it's the mean per family; error bars would be 0.
    """
    # aggregate mean per family
    fams = sorted(set(r["family"] for r in results))
    means = []
    for fam in fams:
        vals = [r["recon"] for r in results if r["family"] == fam and np.isfinite(r["recon"])]
        means.append(np.mean(vals) if len(vals) else np.nan)

    plt.figure(figsize=(7, 4))
    x = np.arange(len(fams))
    plt.bar(x, means)
    plt.xticks(x, fams, rotation=0)
    plt.ylabel("Reconstruction loss (mean over variants)")
    plt.title(title)
    plt.tight_layout()

# ---------- Helpers ----------
def flatten_cols(df):
    """('distill_loss','mean') -> 'distill_loss_mean'."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ['_'.join([str(c) for c in tup if c!='']).strip('_') for tup in df.columns]

    return df.reset_index().rename(columns={
        'config/activation':'activation',
        'config/bottleneck_activation':'bottleneck_activation',
        'config/hidden_dims':'hidden_dims',
        'distill_loss_mean':'distill_loss',
        'recon_loss_mean':'recon_loss',
        'time_total_s_mean':'time_total_s',
    })

def infer_arch_cols(df):
    def _len_dims(s):
        try:
            if isinstance(s, (list, tuple)): return len(s)
            return len(ast.literal_eval(str(s)))
        except Exception:
            # fallback: count numbers inside brackets
            return len(re.findall(r'-?\d+\.?\d*', str(s)))
    def _signature(s):
        try:
            xs = ast.literal_eval(str(s))
            return '×'.join(map(str, xs))
        except Exception:
            return str(s)
    df = df.copy()
    df['arch_depth'] = df['hidden_dims'].map(_len_dims)       # e.g., 4, 5
    df['arch_sig']   = df['hidden_dims'].map(_signature)      # e.g., 1500×1000×500×250
    return df

def prep(df):
    df = flatten_cols(df)
    # keep only rows with the means we need
    for c in ['teacher','activation','bottleneck_activation','hidden_dims','distill_loss','recon_loss']:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df = infer_arch_cols(df)
    # Clean categories
    df['bottleneck_activation'] = df['bottleneck_activation'].fillna('None').replace({'nan':'None'})
    df['activation'] = df['activation'].replace({'ReLU':'ReLU', 'SELU':'SELU'})
    return df

def make_depth_figs(df, task_name="task"):
    df = prep(df).copy()

    # --- Fig: Architecture sensitivity on distill loss ---
    # Boxplot across architectures, colored by teacher. Depth is compact, signature is more detailed.
    plt.figure(figsize=(9,4.8))
    sns.barplot(
        data=df,
        x='arch_depth', y='distill_loss', hue='teacher',
        estimator='mean', errorbar=('ci', 95)
    )
    plt.yscale('log')
    plt.xlabel('Architecture depth (# hidden layers)')
    plt.ylabel('Distill loss (mean, log scale)')
    plt.title(f'{task_name}: Distill loss vs architecture depth')
    plt.tight_layout()
    plt.show()

def make_activation_figs(df, task_name="task"):
    df = prep(df).copy()

    # --- Fig A: Distill vs bottleneck activation (by teacher) ---
    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(2, 2, 1)
    order = ['None','ReLU','SELU'] if set(df['bottleneck_activation']) >= {'None','ReLU','SELU'} else None
    hue = 'teacher' if df['teacher'].nunique() > 1 else None
    dodge = 0.25 if hue else False
    sns.pointplot(
        data=df,
        x='bottleneck_activation', y='distill_loss', hue='teacher',
        order=order, errorbar=('ci',95), dodge=dodge, ax = ax1
    )
    ax1.set_yscale('log')             
    ax1.set_xlabel('Bottleneck activation')
    ax1.set_ylabel('Distill loss (mean, log scale)')
    ax1.set_title(f'{task_name}: Distill loss vs bottleneck (lower is better)')

    # --- Fig B: Recon vs hidden-layer activation (by teacher) ---
    ax2 = fig.add_subplot(2, 2, 2)
    sns.pointplot(
        data=df,
        x='activation', y='recon_loss', hue='teacher',
        errorbar=('ci',95), dodge=dodge, ax = ax2
    )
    ax2.set_xlabel('Hidden-layer activation')
    ax2.set_ylabel('Recon loss (mean)')
    ax2.set_title(f'{task_name}: Reconstruction vs hidden-layer activation')

    # Optional: scatter to see both effects at once (color=bottleneck, marker=activation)
    ax3 = fig.add_subplot(2, 2, (3,4))
    mstyles = {'ReLU':'o','SELU':'s'}
    for (bn, act), g in df.groupby(['bottleneck_activation','activation']):
        ax3.scatter(g['recon_loss'], g['distill_loss'],
                    label=f'bn={bn}, act={act}', marker=mstyles.get(act,'o'), alpha=0.5)
    ax3.set_yscale('log')
    ax3.set_xlabel('Recon loss (mean)')
    ax3.set_ylabel('Distill loss (mean, log scale)')
    ax3.set_title(f'{task_name}: Recon vs Distill by activations')
    ax3.legend(frameon=False, ncol=2, title='Activation',loc='lower right', bbox_to_anchor=(1.52, 0.2))
    plt.tight_layout()
    plt.show()

def _class_colors():
    # 10 distinct colors for digits 0..9 (matplotlib tab10)
    cmap = plt.get_cmap("tab10")
    return {d: cmap(d) for d in range(10)}

def plot_numbers(ax, Z, y, text_subset=5000, fontsize=6, alpha=0.9):
    """
    Plots digits as text at their embedding coordinates.
    Uses a stratified subsample for readability/speed.
    """
    rng = np.random.default_rng(0)
    colors = _class_colors()

    # stratified subsampling per digit
    idxs = []
    per_class = max(1, text_subset // 10)
    for d in range(10):
        d_idx = np.where(y == d)[0]
        take = min(per_class, len(d_idx))
        if take > 0:
            choice = rng.choice(d_idx, size=take, replace=False)
            idxs.append(choice)
    if idxs:
        idxs = np.concatenate(idxs)
    else:
        idxs = np.arange(min(text_subset, len(y)))

    for i in idxs:
        ax.text(
            Z[i, 0],
            Z[i, 1],
            str(int(y[i])),
            color=colors[int(y[i])],
            fontsize=fontsize,
            alpha=alpha,
            ha="center",
            va="center",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")