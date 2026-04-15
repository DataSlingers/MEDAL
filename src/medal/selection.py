"""
Teacher-hyperparameter selection and visualisation utilities.

Typical usage::

    from medal.selection import select_teacher_param, plot_reconstruction_error

    df = results.load_metrics(X_train, X_val, X_test)
    opt_param = select_teacher_param(df, param_col="perplexity")
    plot_reconstruction_error(df, opt_param, param_col="perplexity")
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Plot style constants (applied lazily so importing this module does
# not require matplotlib to be available at import time).
# ------------------------------------------------------------------

_RCPARAMS = {
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size":            11,
    "axes.labelsize":       11,
    "axes.titlesize":       12,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "axes.linewidth":       0.6,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "lines.linewidth":      1.2,
    "lines.markersize":     3.5,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
}

_LINE_COLOR  = "#C0446A"
_BOX_COLOR   = "#4A90C4"
_LEFT_WSPACE = 0.25


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def select_teacher_param(
    df: pd.DataFrame,
    param_col: str,
    metric_col: str = "recon_loss",
    val_split: str = "Val",
) -> object:
    """
    Choose the smallest teacher hyperparameter whose mean validation
    reconstruction loss lies within one SEM of the global minimum.

    This is the "one-SEM rule": among all configs whose mean val loss
    is within ``[best_mean - best_sem, best_mean + best_sem]``, prefer
    the smallest (most conservative) parameter value.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :meth:`~medal.sweep.SweepResults.load_metrics`, containing
        columns ``[param_col, "split", metric_col]``.
    param_col : str
        Column name of the swept hyperparameter (e.g. ``"perplexity"``).
    metric_col : str
        Loss column to minimise (default ``"recon_loss"``).
    val_split : str
        Value in the ``"split"`` column to use for selection.

    Returns
    -------
    opt_param : scalar
        The selected hyperparameter value.
    """
    val_df = df[df["split"] == val_split]
    stats = val_df.groupby(param_col)[metric_col].agg(["mean", "sem"])
    argmin = stats["mean"].idxmin()
    best_mean, best_sem = stats.loc[argmin, ["mean", "sem"]]
    threshold = best_mean + best_sem
    candidates = stats[stats["mean"] <= threshold]
    return candidates.index.min()


def plot_reconstruction_error(
    df: pd.DataFrame,
    opt_param,
    param_col: str,
    emb_data: Optional[dict] = None,
    params_to_show: Optional[List] = None,
    palette=None,
    param_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 7),
    show_boxplot: bool = True,
    row_label_y: float = 1.15,
    legend_kw: Optional[dict] = None,
    xtick_fontsize: int = 10,
    share_x_lines: bool = True,
) -> "plt.Figure":
    """
    Combined reconstruction-error panel with optional embedding grid.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :meth:`~medal.sweep.SweepResults.load_metrics`.
    opt_param : scalar
        Optimal parameter value returned by :func:`select_teacher_param`.
    param_col : str
        Column name of the swept hyperparameter.
    emb_data : dict, optional
        ``{param_val: {"Train": (Z, recon_errors, labels), ...}}``.
        When provided, a label-coloured embedding grid is shown on the right.
    params_to_show : list, optional
        Subset of parameter values to show in the embedding grid.
    palette : optional
        Colour palette passed to seaborn.
    param_label : str, optional
        Display name for the parameter axis (defaults to ``param_col``).
    xlabel : str, optional
        x-axis label for the line/box panel (defaults to ``param_col``).
    figsize : tuple
    show_boxplot : bool
        Show median boxplot next to the mean line panel.
    row_label_y, legend_kw, xtick_fontsize, share_x_lines
        Forwarded to :func:`_plot_line_box_panel`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update(_RCPARAMS)

    df = df.sort_values(by=param_col)
    unique_params = sorted(df[param_col].unique())

    val_df = df[df["split"] == "Val"]
    stats = val_df.groupby(param_col)["recon_loss"].agg(["mean", "sem"])
    argmin = stats["mean"].idxmin()
    best_mean, best_sem = stats.loc[argmin, ["mean", "sem"]]
    one_std_range = (best_mean - best_sem, best_mean + best_sem)
    vline_x = unique_params.index(opt_param)

    xlabel = xlabel or param_col
    param_label = param_label or param_col

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.10)

    if emb_data is not None:
        gs = GridSpec(1, 2, figure=fig,
                      width_ratios=[1.2 if show_boxplot else 0.7, 2.5],
                      wspace=0.15)
        _plot_line_box_panel(
            fig, gs[0, 0], df, param_col, unique_params,
            one_std_range, vline_x, xlabel,
            row_label_y=row_label_y, legend_kw=legend_kw,
            xtick_fontsize=xtick_fontsize,
            share_x_lines=share_x_lines,
            show_boxplot=show_boxplot,
        )
        _plot_embedding_grid(
            fig, gs[0, 1], emb_data,
            params_to_show or unique_params,
            opt_param, param_label, palette=palette,
        )
    else:
        gs = GridSpec(1, 1, figure=fig)
        _plot_line_box_panel(
            fig, gs[0, 0], df, param_col, unique_params,
            one_std_range, vline_x, xlabel,
            row_label_y=row_label_y, legend_kw=legend_kw,
            xtick_fontsize=xtick_fontsize,
            share_x_lines=share_x_lines,
            show_boxplot=show_boxplot,
        )

    plt.show()
    return fig


def plot_distortion_map(
    emb_data: dict,
    params_to_show: List,
    opt_param,
    param_label: str,
    cmap="magma",
    colorbar_label: str = "Recon. MSE",
    fig_title: str = "Reconstruction error over embedding",
    figsize: Tuple[int, int] = (9, 6),
) -> "plt.Figure":
    """
    Standalone distortion-map figure: embeddings coloured by per-point
    reconstruction error instead of class label.

    Parameters
    ----------
    emb_data : dict
        ``{param_val: {"Train": (Z, recon_errors, labels), ...}}``.
    params_to_show : list
        Subset of parameter values whose rows to show.
    opt_param : scalar
        Optimal parameter (row is annotated as "Optimum").
    param_label : str
        Display name for row labels (e.g. ``"perplexity"``).
    cmap : str or Colormap
    colorbar_label : str
    fig_title : str
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update(_RCPARAMS)

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.12)
    gs = GridSpec(1, 1, figure=fig)
    _plot_embedding_grid(
        fig, gs[0, 0], emb_data, params_to_show,
        opt_param, param_label,
        cmap=cmap, colorbar=True,
        colorbar_label=colorbar_label,
        fig_title=fig_title,
        row_label_fontsize=10,
    )
    plt.show()
    return fig


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _plot_line_box_panel(
    fig, gs_element, analysis: pd.DataFrame, param_col: str,
    unique_params: List, one_std_range: Tuple[float, float], vline_x: int,
    xlabel: str,
    line_color: str = _LINE_COLOR,
    box_color: str  = _BOX_COLOR,
    left_wspace: float = _LEFT_WSPACE,
    row_label_y: float = 1.15,
    legend_kw: Optional[dict] = None,
    xtick_fontsize: int = 10,
    share_x_lines: bool = True,
    show_boxplot: bool = True,
):
    """Line (mean ± SEM) and optional boxplot (median) panels."""
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    import seaborn as sns

    if legend_kw is None:
        legend_kw = {"frameon": False, "fontsize": 11, "loc": "upper left"}

    gs_kw   = {"wspace": left_wspace} if show_boxplot else {}
    gs_left = gs_element.subgridspec(3, 2 if show_boxplot else 1, hspace=0.45, **gs_kw)

    ax_line_train = fig.add_subplot(gs_left[0, 0])
    share_kw      = {"sharex": ax_line_train} if share_x_lines else {}
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

    for ax_line, ax_box, split in zip(line_axes, box_axes, ["Train", "Val", "Test"]):
        subset = analysis[analysis["split"] == split]

        sns.pointplot(data=subset, x=param_col, y="recon_loss",
                      errorbar="se", color=line_color, ax=ax_line,
                      markersize=3, linewidth=1.2, err_kws={"linewidth": 0.6})
        sns.stripplot(data=subset, x=param_col, y="recon_loss",
                      color=line_color, size=1.8, alpha=0.3, ax=ax_line, zorder=0)

        if ax_box is not None:
            sns.boxplot(data=subset, x=param_col, y="recon_loss",
                        color=box_color, ax=ax_box, linewidth=0.6,
                        flierprops=dict(marker="o", markersize=1.5,
                                        markerfacecolor=box_color, alpha=0.4))

        for ax in [a for a in (ax_line, ax_box) if a is not None]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_linewidth(0.6)
            ax.grid(axis="y", linewidth=0.35, color="gray", alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=6)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

        if ax_box is not None:
            ax_box.set_ylabel("")
            ax_box.tick_params(axis="y", labelleft=False)

        ax_line.text(row_label_x, row_label_y, split,
                     transform=ax_line.transAxes,
                     ha="center", va="center",
                     fontsize=12, fontweight="bold", clip_on=False)

        if split == "Val":
            ax_line.set_ylabel("Reconstruction Loss", labelpad=2,
                                fontsize=20, fontweight="bold")
            ax_line.fill_between(range(-1, len(unique_params) + 1),
                                  *one_std_range, color=line_color, alpha=0.12,
                                  label="1 SEM")
            ax_line.axvline(x=vline_x, ls="--", lw=0.8, color="#333333", zorder=5)
            ax_line.legend(**legend_kw)
            ax_line.set_xlim(-0.5, len(unique_params) - 0.5)
            if ax_box is not None:
                ax_box.axvline(x=vline_x, ls="--", lw=0.8, color="#333333", zorder=5)

    if show_boxplot:
        ax_line_train.text(0.5, 1.15, "Mean",
                           transform=ax_line_train.transAxes,
                           ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax_box_train.text(0.5, 1.15, "Median",
                          transform=ax_box_train.transAxes,
                          ha="center", va="bottom", fontsize=10, fontweight="bold")

    for ax in [ax_line_train, ax_line_val] + ([ax_box_train, ax_box_val] if show_boxplot else []):
        plt.setp(ax.get_xticklabels(), visible=False)

    for ax in [ax_line_test] + ([ax_box_test] if show_boxplot else []):
        plt.setp(ax.get_xticklabels(), rotation=45, ha="center", fontsize=xtick_fontsize)
        ax.set_xlabel(xlabel, labelpad=2, fontsize=10)


def _plot_embedding_grid(
    fig, gs_element, emb_data: dict, params_to_show: List,
    opt_param, param_label: str,
    palette=None,
    cmap=None,
    colorbar: bool = False,
    colorbar_label: str = "Recon. MSE",
    fig_title: Optional[str] = None,
    row_label_fontsize: int = 12,
):
    """
    Unified embedding grid for label-coloured and distortion-map plots.

    colorbar=False  -> colour by class label via seaborn  (palette required)
    colorbar=True   -> colour by recon loss via scatter + per-row colorbar
                       (cmap required)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    col_titles = ["Train", "Val", "Test"]
    n_rows     = len(params_to_show)

    gs = gs_element.subgridspec(n_rows, 4,
                                width_ratios=[1, 1, 1, 0.07],
                                hspace=0.15, wspace=0.08)
    emb_axes = np.array([
        [fig.add_subplot(gs[row, col]) for col in range(3)]
        for row in range(n_rows)
    ])

    for row, param in enumerate(params_to_show):
        data = {}
        for s in col_titles:
            Z, recon, labs = emb_data[param][s]
            data[s] = (
                Z.detach().numpy() if hasattr(Z, "detach") else np.array(Z),
                recon,
                labs,
            )

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

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

            if row == 0:
                ax.set_title(split, fontsize=12, fontweight="bold", pad=3)

            if col == 0:
                label = (f"Optimum ({param_label}={param})"
                         if param == opt_param else f"{param_label}={param}")
                bbox = ax.get_position()
                fig.text(bbox.x0 - 0.01, bbox.y0 + bbox.height / 2,
                         label, ha="right", va="center",
                         fontsize=row_label_fontsize, fontweight="bold", rotation=90)

        if colorbar:
            cb = fig.colorbar(sc, cax=cax)
            cb.ax.tick_params(labelsize=8, width=0.4, length=2)
            cb.outline.set_linewidth(0.4)
            cb.set_label(colorbar_label, fontsize=9, labelpad=2)

    if fig_title is not None:
        fig.text(0.5, 0.96, fig_title,
                 ha="center", va="bottom", fontsize=12, fontweight="bold")
