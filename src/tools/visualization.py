"""Matplotlib visualizers for belief states and utterances.

Two primitives, both returning a ``matplotlib.figure.Figure`` (so the caller decides
whether to ``show`` / ``savefig``):

  * ``plot_belief_states`` -- a grid of bar charts, one per categorical belief.
  * ``plot_utterances``    -- a grid of rendered utterance images.

Both accept a single item or an arbitrarily-batched array (any leading axes -- e.g.
``[num_agents, batch, ...]`` -- are flattened into the grid). They take plain numpy or
jax arrays; nothing here is jitted, so it is inspection-only glue, not part of the
training graph.
"""

from __future__ import annotations
import math
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tools.utterance_rendering import paint_multiple_splines


def _grid(num_items: int, ncols: Optional[int], item_size: float = 2.5):
    """Make a (fig, flat-list-of-axes) grid sized to hold ``num_items`` panels."""
    ncols = ncols or min(num_items, 4)
    nrows = math.ceil(num_items / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * item_size, nrows * item_size), squeeze=False)
    flat_axes = axes.flatten()
    # Hide any trailing panels the item count doesn't fill.
    for ax in flat_axes[num_items:]:
        ax.axis("off")
    return fig, flat_axes


def _flatten_leading(arr: np.ndarray, item_ndim: int) -> np.ndarray:
    """Collapse every leading (batch) axis into one, keeping the last ``item_ndim`` axes."""
    if arr.ndim < item_ndim:
        raise ValueError(f"expected at least {item_ndim} dims, got shape {arr.shape}")
    if arr.ndim == item_ndim:
        arr = arr[None]
    return arr.reshape((-1, *arr.shape[-item_ndim:]))


def plot_belief_states(
    beliefs,
    titles: Optional[Sequence[str]] = None,
    ncols: Optional[int] = None,
    fig_title: Optional[str] = None,
    state_labels: Optional[Sequence[str]] = None,
) -> Figure:
    """Bar-chart one or more categorical belief distributions.

    Args:
        beliefs: ``[belief_dim]`` or any batched ``[..., belief_dim]`` array of
            distributions over world states; leading axes are flattened into the grid.
        titles: Optional per-panel titles (in flattened order).
        ncols: Panels per row (default: up to 4).
        fig_title: Optional suptitle.
        state_labels: Optional x-tick labels for the ``belief_dim`` states.

    Returns:
        The Figure (nothing is shown/saved here).
    """
    flat = _flatten_leading(np.asarray(beliefs, dtype=float), item_ndim=1)
    num_items, belief_dim = flat.shape

    fig, axes = _grid(num_items, ncols)
    states = np.arange(belief_dim)
    for i, ax in enumerate(axes[:num_items]):
        ax.bar(states, flat[i], color="#4c72b0")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(states)
        if state_labels is not None:
            ax.set_xticklabels(state_labels, rotation=45, ha="right", fontsize=7)
        else:
            ax.tick_params(labelsize=7)
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_utterances(
    utterances,
    image_dim: int = 32,
    render: Optional[bool] = None,
    titles: Optional[Sequence[str]] = None,
    ncols: Optional[int] = None,
    fig_title: Optional[str] = None,
) -> Figure:
    """Show one or more utterances as images.

    Accepts either flat spline-parameter vectors (rendered here via
    ``paint_multiple_splines``) or pre-rendered ``(image_dim, image_dim)`` canvases, with
    any leading batch axes. By default it auto-detects which was passed -- an input whose
    last two axes are both ``image_dim`` is treated as already rendered -- but ``render``
    can force it either way.

    Args:
        utterances: ``[..., utterance_action_dim]`` spline params, or ``[..., image_dim,
            image_dim]`` rendered canvases. Leading axes are flattened into the grid.
        image_dim: Canvas side length (used to render, and for auto-detection).
        render: Force rendering (``True``) or force treating input as images (``False``);
            ``None`` auto-detects.
        titles: Optional per-panel titles (in flattened order).
        ncols: Panels per row (default: up to 4).
        fig_title: Optional suptitle.

    Returns:
        The Figure (nothing is shown/saved here).
    """
    arr = np.asarray(utterances)

    if render is None:
        already_rendered = arr.ndim >= 2 and arr.shape[-1] == image_dim and arr.shape[-2] == image_dim
        render = not already_rendered

    if render:
        params = _flatten_leading(arr.astype(float), item_ndim=1)  # [K, utterance_action_dim]
        images = np.asarray(paint_multiple_splines(params, image_dim))  # [K, image_dim, image_dim]
    else:
        images = _flatten_leading(arr, item_ndim=2)  # [K, H, W]

    num_items = images.shape[0]
    fig, axes = _grid(num_items, ncols)
    for i, ax in enumerate(axes[:num_items]):
        ax.imshow(images[i], cmap="gray_r", vmin=0.0, vmax=1.0, origin="upper")
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import jax

    key = jax.random.key(0)
    b_key, u_key = jax.random.split(key)

    # Six random belief distributions over 4 states.
    beliefs = jax.random.dirichlet(b_key, alpha=np.ones(4), shape=(6,))
    belief_fig = plot_belief_states(
        beliefs,
        titles=[f"agent {i}" for i in range(6)],
        fig_title="Belief states (demo)",
        state_labels=["s0", "s1", "s2", "done"],
    )
    belief_fig.savefig("belief_states_demo.png", dpi=150)

    # Six random utterances (2 splines = 12 params each), rendered to 64x64 canvases.
    utterances = jax.random.uniform(u_key, (6, 12))
    utt_fig = plot_utterances(
        utterances,
        image_dim=64,
        titles=[f"agent {i}" for i in range(6)],
        fig_title="Utterances (demo)",
    )
    utt_fig.savefig("utterances_demo.png", dpi=150)

    print("Saved belief_states_demo.png and utterances_demo.png")
