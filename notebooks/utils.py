import numpy as np
from typing import Tuple, Dict, Optional, Union
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def _ensure_images_shape(X: np.ndarray) -> Tuple[np.ndarray, bool]:
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[1] == 784:
        X_img = X.reshape(-1, 28, 28)
        was_flattened = True
    elif X.ndim == 3 and X.shape[1:] == (28, 28):
        X_img = X
        was_flattened = False
    else:
        raise ValueError("X must be shape (N,784) or (N,28,28).")
    X_img = X_img.astype(np.float32)
    if X_img.max() > 1.0:
        X_img = X_img / 255.0
    X_img = np.clip(X_img, 0.0, 1.0)
    return X_img, was_flattened

def _restore_shape(X_img: np.ndarray, was_flattened: bool) -> np.ndarray:
    return X_img.reshape(len(X_img), -1) if was_flattened else X_img

# -------------------------
# Mask helpers
# -------------------------
def _normalize_mask(
    pixel_mask: Optional[np.ndarray],
    mask_frac: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Returns a boolean mask of shape (28,28), or None meaning "all pixels".
    If pixel_mask is provided, it is converted to boolean and reshaped if needed.
    If not provided, a random mask is generated using mask_frac.
    """
    if pixel_mask is None:
        if mask_frac >= 1.0:
            return None  # treat as "all pixels"
        if not (0.0 < mask_frac < 1.0):
            raise ValueError("mask_frac must be in (0,1] when pixel_mask is None.")
        m = rng.random((28, 28)) < mask_frac
        return m.astype(bool)

    m = np.asarray(pixel_mask)
    if m.shape == (784,):
        m = m.reshape(28, 28)
    if m.shape != (28, 28):
        raise ValueError("pixel_mask must have shape (28,28) or (784,).")
    return m.astype(bool)

def _maybe_per_image_mask(
    base_mask: Optional[np.ndarray],
    mask_mode: str,
    mask_frac: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    If mask_mode == 'per_image', ignore base_mask and sample a new mask.
    If 'shared', return base_mask as-is (could be None meaning all pixels).
    """
    if mask_mode not in ("shared", "per_image"):
        raise ValueError("mask_mode must be 'shared' or 'per_image'.")
    if mask_mode == "shared":
        return base_mask
    # per_image
    if mask_frac >= 1.0:
        return None
    m = rng.random((28, 28)) < mask_frac
    return m.astype(bool)

# -------------------------
# Masked transforms
# -------------------------
def _apply_noise(img: np.ndarray, sigma: float, rng: np.random.Generator, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
        return np.clip(img + noise, 0.0, 1.0)
    out = img.copy()
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    out[mask] = out[mask] + noise[mask]
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _apply_brightness_contrast(img: np.ndarray, alpha: float, beta: float, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return np.clip(alpha * img + beta, 0.0, 1.0).astype(np.float32)
    out = img.copy()
    out[mask] = alpha * out[mask] + beta
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _apply_smoothing(img: np.ndarray, n_iter: int = 1, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute smoothing update as before, but only write back into masked pixels.
    Note: the blur uses full neighborhood context (including unmasked pixels),
    but only masked pixels are changed.
    """
    if mask is None:
        out = img.copy()
        for _ in range(n_iter):
            up    = np.roll(out, shift=-1, axis=0)
            down  = np.roll(out, shift=+1, axis=0)
            left  = np.roll(out, shift=-1, axis=1)
            right = np.roll(out, shift=+1, axis=1)
            out = (out + up + down + left + right) / 5.0
        return out.astype(np.float32)

    out = img.copy()
    tmp = out.copy()
    for _ in range(n_iter):
        up    = np.roll(tmp, shift=-1, axis=0)
        down  = np.roll(tmp, shift=+1, axis=0)
        left  = np.roll(tmp, shift=-1, axis=1)
        right = np.roll(tmp, shift=+1, axis=1)
        blurred = (tmp + up + down + left + right) / 5.0
        # only update masked pixels
        tmp = tmp.copy()
        tmp[mask] = blurred[mask]
    out = tmp
    return out.astype(np.float32)

def _make_checkerboard(h: int = 28, w: int = 28, amp: float = 0.10) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    pat = ((xx + yy) % 2) * 2 - 1
    return (amp * pat).astype(np.float32)

def _apply_checker_watermark(
    img: np.ndarray,
    pattern: np.ndarray,
    clip: bool = True,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if mask is None:
        out = img + pattern
    else:
        out = img.copy()
        out[mask] = out[mask] + pattern[mask]
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)

# -------------------------
# Main function
# -------------------------
def make_artificial_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_fracs: Dict[str, float] = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
    noise_sigma: float = 0.20,
    alpha: float = 1.20,
    beta: float = 0.05,
    smooth_iters: int = 50,
    checker_amp: float = 0.10,
    checker_batch: str = "D",
    checker_clip: bool = True,
    seed: int = 0,
    stratify_by_label: bool = True,
    # --- NEW: pixel subset controls ---
    pixel_mask: Optional[np.ndarray] = None,   # (28,28) or (784,); True=can change
    mask_frac: float = 1.0,                    # used only if pixel_mask is None OR mask_mode="per_image"
    mask_mode: str = "shared",                 # "shared" or "per_image"
    mask_seed: int = 0,                        # separate seed for mask generation
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(sum(batch_fracs.values()) - 1.0) < 1e-6, "batch_fracs must sum to 1.0"
    X_img, was_flattened = _ensure_images_shape(X)
    y = np.asarray(y)
    assert len(X_img) == len(y), "X and y must have same length"

    rng = np.random.default_rng(seed)
    rng_mask = np.random.default_rng(mask_seed)

    N = len(X_img)
    batch_ids = np.empty(N, dtype=object)

    def assign_indices(idxs: np.ndarray, fracs: Dict[str, float]) -> Dict[str, np.ndarray]:
        n = len(idxs)
        counts = {k: int(round(fracs[k] * n)) for k in fracs}
        diff = n - sum(counts.values())
        if diff != 0:
            k_adj = max(fracs, key=fracs.get)
            counts[k_adj] += diff
        rng.shuffle(idxs)
        splits = {}
        start = 0
        for k in fracs:
            end = start + counts[k]
            splits[k] = idxs[start:end]
            start = end
        return splits

    if stratify_by_label:
        for digit in np.unique(y):
            idxs = np.where(y == digit)[0]
            splits = assign_indices(idxs, batch_fracs)
            for k, ss in splits.items():
                batch_ids[ss] = k
    else:
        idxs = np.arange(N)
        splits = assign_indices(idxs, batch_fracs)
        for k, ss in splits.items():
            batch_ids[ss] = k

    if checker_batch not in batch_fracs:
        raise ValueError(f"checker_batch={checker_batch!r} must be one of {list(batch_fracs.keys())}")
    checker_pattern = _make_checkerboard(28, 28, amp=checker_amp)

    # base mask (used if mask_mode="shared")
    base_mask = _normalize_mask(pixel_mask, mask_frac=mask_frac, rng=rng_mask)

    X_out = np.empty_like(X_img, dtype=np.float32)
    for b in ["A", "B", "C", "D"]:
        idxs = np.where(batch_ids == b)[0]
        if len(idxs) == 0:
            continue

        for ii in idxs:
            im = X_img[ii]
            mask = _maybe_per_image_mask(base_mask, mask_mode=mask_mode, mask_frac=mask_frac, rng=rng_mask)

            if b == checker_batch:
                X_out[ii] = _apply_checker_watermark(im, checker_pattern, clip=checker_clip, mask=mask)
            elif b == "A":
                X_out[ii] = im
            elif b == "B":
                X_out[ii] = _apply_noise(im, noise_sigma, rng, mask=mask)
            elif b == "C":
                X_out[ii] = _apply_brightness_contrast(im, alpha, beta, mask=mask)
            elif b == "D":
                X_out[ii] = _apply_smoothing(im, n_iter=smooth_iters, mask=mask)
            else:
                raise ValueError(f"Unknown batch key {b}")

    X_aug = _restore_shape(X_out, was_flattened)
    y_bio = y.copy()
    y_batch = batch_ids.astype(str)
    return X_aug, y_bio, y_batch


# ── Reconstruction-analysis helper functions ──────────────────────────────────
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np, pandas as pd

plt.rcParams.update({
    'font.family':          'sans-serif',
    'font.sans-serif':      ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size':            11,
    'axes.labelsize':       11,
    'axes.titlesize':       12,
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'axes.linewidth':       0.6,
    'xtick.major.width':    0.6,
    'ytick.major.width':    0.6,
    'xtick.major.size':     2.5,
    'ytick.major.size':     2.5,
    'lines.linewidth':      1.2,
    'lines.markersize':     3.5,
    'pdf.fonttype':         42,
    'ps.fonttype':          42,
})

LINE_COLOR  = '#C0446A'
BOX_COLOR   = '#4A90C4'
LEFT_WSPACE = 0.25


def compute_param_choice(res_dict, param_col):
    analysis = pd.DataFrame(res_dict)
    analysis.sort_values(by=param_col, inplace=True)

    test_stats = analysis[analysis.split == 'Val'].groupby(param_col).agg(
        mean=('recon_loss', 'mean'),
        sem=('recon_loss', 'sem'),
    )

    argmin_mean          = test_stats['mean'].idxmin()
    best_mean, best_sem  = test_stats.loc[argmin_mean, ['mean', 'sem']]
    one_std_range        = (best_mean - best_sem, best_mean + best_sem)

    opt_param = test_stats.loc[
        (test_stats['mean'] <= one_std_range[1]) &
        (test_stats['mean'] >= one_std_range[0])
    ].index.min()

    unique_params = sorted(analysis[param_col].unique())
    vline_x       = unique_params.index(opt_param)

    return analysis, one_std_range, opt_param, unique_params, vline_x


def plot_line_box_panel(fig, gs_element, analysis, param_col,
                        unique_params, one_std_range, vline_x, xlabel,
                        line_color=LINE_COLOR, box_color=BOX_COLOR,
                        left_wspace=LEFT_WSPACE, row_label_y=1.15,
                        legend_kw=None, xtick_fontsize=10,
                        share_x_lines=True, show_boxplot=True):
    """
    Plot line (mean +/- SEM) and optional boxplot (median) panels.

    show_boxplot=False renders only the line column; row labels are then
    centred directly above each line-plot row.
    """
    if legend_kw is None:
        legend_kw = {'frameon': False, 'fontsize': 11, 'loc': 'upper left'}

    gs_kw   = {'wspace': left_wspace} if show_boxplot else {}
    gs_left = gs_element.subgridspec(3, 2 if show_boxplot else 1, hspace=0.45, **gs_kw)

    ax_line_train = fig.add_subplot(gs_left[0, 0])
    share_kw      = {'sharex': ax_line_train} if share_x_lines else {}
    ax_line_val   = fig.add_subplot(gs_left[1, 0], **share_kw)
    ax_line_test  = fig.add_subplot(gs_left[2, 0], **share_kw)
    line_axes     = [ax_line_train, ax_line_val, ax_line_test]

    if show_boxplot:
        ax_box_train = fig.add_subplot(gs_left[0, 1], sharey=ax_line_train)
        ax_box_val   = fig.add_subplot(gs_left[1, 1], sharex=ax_box_train, sharey=ax_line_val)
        ax_box_test  = fig.add_subplot(gs_left[2, 1], sharex=ax_box_train, sharey=ax_line_test)
        box_axes = [ax_box_train, ax_box_val, ax_box_test]
    else:
        box_axes = [None, None, None]

    row_label_x = 1 + left_wspace / 2 if show_boxplot else 0.5

    for ax_line, ax_box, split in zip(line_axes, box_axes, ['Train', 'Val', 'Test']):
        subset = analysis[analysis.split == split]

        sns.pointplot(data=subset, x=param_col, y='recon_loss',
                      errorbar='se', color=line_color, ax=ax_line,
                      markersize=3, linewidth=1.2, err_kws={'linewidth': 0.6})
        sns.stripplot(data=subset, x=param_col, y='recon_loss',
                      color=line_color, size=1.8, alpha=0.3, ax=ax_line, zorder=0)

        if ax_box is not None:
            sns.boxplot(data=subset, x=param_col, y='recon_loss',
                        color=box_color, ax=ax_box, linewidth=0.6,
                        flierprops=dict(marker='o', markersize=1.5,
                                        markerfacecolor=box_color, alpha=0.4))

        for ax in [a for a in (ax_line, ax_box) if a is not None]:
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_linewidth(0.6)
            ax.grid(axis='y', linewidth=0.35, color='gray', alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', labelsize=6)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

        if ax_box is not None:
            ax_box.set_ylabel('')
            ax_box.tick_params(axis='y', labelleft=False)

        ax_line.text(row_label_x, row_label_y, split,
                     transform=ax_line.transAxes,
                     ha='center', va='center',
                     fontsize=12, fontweight='bold', clip_on=False)

        if split == 'Val':
            ax_line.set_ylabel('Reconstruction Loss', labelpad=2, fontsize=20, fontweight='bold')
            ax_line.fill_between(range(-1, len(unique_params) + 1),
                                 *one_std_range, color=line_color, alpha=0.12, label='1 SEM')
            ax_line.axvline(x=vline_x, ls='--', lw=0.8, color='#333333', zorder=5)
            ax_line.legend(**legend_kw)
            ax_line.set_xlim(-0.5, len(unique_params) - 0.5)
            if ax_box is not None:
                ax_box.axvline(x=vline_x, ls='--', lw=0.8, color='#333333', zorder=5)

    if show_boxplot:
        ax_line_train.text(0.5, 1.15, 'Mean',
                       transform=ax_line_train.transAxes,
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax_box_train.text(0.5, 1.15, 'Median',
                          transform=ax_box_train.transAxes,
                          ha='center', va='bottom', fontsize=10, fontweight='bold')

    for ax in [ax_line_train, ax_line_val] + ([ax_box_train, ax_box_val] if show_boxplot else []):
        plt.setp(ax.get_xticklabels(), visible=False)

    for ax in [ax_line_test] + ([ax_box_test] if show_boxplot else []):
        plt.setp(ax.get_xticklabels(), rotation=45, ha='center', fontsize=xtick_fontsize)
        ax.set_xlabel(xlabel, labelpad=2, fontsize=10)


def plot_embedding_grid(fig, gs_element, emb_data, params_to_show,
                        opt_param, param_label,
                        palette=None, cmap=None,
                        colorbar=False, colorbar_label='Recon. MSE',
                        fig_title=None, row_label_fontsize=12):
    """
    Unified embedding grid for label-coloured and distortion-map plots.

    colorbar=False  ->  colour by class label via seaborn  (palette required)
    colorbar=True   ->  colour by recon loss via scatter + per-row colorbar
                        (cmap required)
    """
    col_titles = ['Train', 'Val', 'Test']
    n_rows     = len(params_to_show)

    gs = gs_element.subgridspec(n_rows, 4,
                                width_ratios=[1, 1, 1, 0.07],
                                hspace=0.15, wspace=0.08)
    emb_axes = np.array([
        [fig.add_subplot(gs[row, col]) for col in range(3)]
        for row in range(n_rows)
    ])

    for row, param in enumerate(params_to_show):
        # Pre-extract all splits; compute shared norm for distortion mode
        data = {}
        for s in col_titles:
            Z, recon, labs = emb_data[param][s]
            data[s] = (Z.detach().numpy() if hasattr(Z, 'detach') else np.array(Z),
                       recon, labs)

        if colorbar:
            all_recon = np.concatenate([data[s][1] for s in col_titles])
            norm      = plt.Normalize(vmin=0, vmax=np.quantile(all_recon, 0.975))
            cax       = fig.add_subplot(gs[row, 3])

        for col, split in enumerate(col_titles):
            ax             = emb_axes[row, col]
            Z, recon, labs = data[split]

            if colorbar:
                sc = ax.scatter(Z[:, 0], Z[:, 1], c=recon, cmap=cmap,
                                norm=norm, s=5, alpha=0.7,
                                linewidths=0, rasterized=True)
            else:
                sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labs, palette=palette,
                                s=4, alpha=0.7, ax=ax, legend=False,
                                linewidths=0, rasterized=True)

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

            if row == 0:
                ax.set_title(split, fontsize=12, fontweight='bold', pad=3)

            if col == 0:
                label = (f'Optimum ({param_label}={param})'
                         if param == opt_param else f'{param_label}={param}')
                bbox = ax.get_position()
                fig.text(bbox.x0 - 0.01, bbox.y0 + bbox.height / 2,
                         label, ha='right', va='center',
                         fontsize=row_label_fontsize, fontweight='bold', rotation=90)

        if colorbar:
            cb = fig.colorbar(sc, cax=cax)
            cb.ax.tick_params(labelsize=8, width=0.4, length=2)
            cb.outline.set_linewidth(0.4)
            cb.set_label(colorbar_label, fontsize=9, labelpad=2)

    if fig_title is not None:
        fig.text(0.5, 0.96, fig_title,
                 ha='center', va='bottom', fontsize=12, fontweight='bold')


def build_figure(analysis, param_col, unique_params, one_std_range, vline_x,
                 emb_data, params_to_show, opt_param, palette, param_label, xlabel,
                 figsize=(15, 7), row_label_y=1.15, legend_kw=None, xtick_fontsize=10,
                 share_x_lines=True, show_boxplot=True):
    """
    Build the combined line/box + label-embedding figure.

    show_boxplot=False narrows the left panel automatically.
    """
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.10)

    gs = GridSpec(1, 2, figure=fig,
                  width_ratios=[1.2 if show_boxplot else 0.7, 2.5],
                  wspace=0.15)

    plot_line_box_panel(fig, gs[0, 0], analysis, param_col,
                        unique_params, one_std_range, vline_x, xlabel,
                        row_label_y=row_label_y, legend_kw=legend_kw,
                        xtick_fontsize=xtick_fontsize,
                        share_x_lines=share_x_lines,
                        show_boxplot=show_boxplot)

    plot_embedding_grid(fig, gs[0, 1], emb_data, params_to_show,
                        opt_param, param_label, palette=palette)

    plt.show()


def build_distortion_figure(emb_data, params_to_show, opt_param, param_label, cmap,
                             colorbar_label='Recon. MSE',
                             fig_title='Reconstruction error over embedding',
                             figsize=(9, 6)):
    """Standalone distortion-map figure (replaces the ad-hoc GridSpec block)."""
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.12)

    gs = GridSpec(1, 1, figure=fig)
    plot_embedding_grid(fig, gs[0, 0], emb_data, params_to_show,
                        opt_param, param_label,
                        cmap=cmap, colorbar=True,
                        colorbar_label=colorbar_label,
                        fig_title=fig_title,
                        row_label_fontsize=10)
    plt.show()