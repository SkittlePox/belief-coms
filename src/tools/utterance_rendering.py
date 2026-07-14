import functools

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


@functools.partial(jax.vmap, in_axes=(0, None))
def paint_multiple_splines(all_spline_params: jnp.array, image_dim: int):
    """Paint multiple splines on a single canvas. Requires utterance_action_dim be a multiple of 6."""

    @jax.vmap
    def paint_spline_on_canvas(spline_params: jnp.array):
        """Paint a single spline on the canvas with specified thickness using advanced indexing."""

        def bezier_spline(t, P0, P1, P2):
            """Compute points on a quadratic Bézier spline for a given t."""
            t = t[:, None]  # Shape (N, 1) to broadcast with P0, P1, P2 of shape (2,)
            P = (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2
            return P  # Returns shape (N, 2), a list of points on the spline

        brush_size = 1

        spline_params *= image_dim

        # P0, P1, P2 = spline_params.reshape((3, 2))
        P0, P1, P2 = spline_params[0:2], spline_params[2:4], spline_params[4:6]
        t_values = jnp.linspace(0, 1, num=50)
        spline_points = bezier_spline(t_values, P0, P1, P2)
        x_points, y_points = jnp.round(spline_points).astype(int).T

        # Generate brush offsets
        brush_offsets = jnp.array(
            [(dx, dy) for dx in range(-brush_size, brush_size) for dy in range(-brush_size, brush_size)]  # brush_size + 1
        )  # brush_size + 1
        x_offsets, y_offsets = brush_offsets.T

        # Calculate all indices to update for each point (broadcasting magic)
        all_x_indices = x_points[:, None] + x_offsets
        all_y_indices = y_points[:, None] + y_offsets

        # Flatten indices and filter out-of-bound ones
        all_x_indices = jnp.clip(all_x_indices.flatten(), 0, image_dim)
        all_y_indices = jnp.clip(all_y_indices.flatten(), 0, image_dim)

        # Update the canvas
        canvas = jnp.zeros((image_dim, image_dim))
        canvas = canvas.at[all_x_indices, all_y_indices].add(1)
        return canvas

    # Vmap over splines and sum contributions
    all_spline_params = jnp.clip(all_spline_params, 0.0, 1.0)
    canvas = jnp.clip(paint_spline_on_canvas(all_spline_params.reshape(-1, 6)).sum(axis=0), 0.0, 1.0)
    return canvas


if __name__ == "__main__":
    IMAGE_DIM = 64

    # Each spline: 6 params [P0_x, P0_y, P1_x, P1_y, P2_x, P2_y] in [0, 1]
    # Outer vmap batches over "utterances", so input shape is (batch, num_splines * 6)
    test_cases = [
        ("Single diagonal", [0.1, 0.1, 0.5, 0.5, 0.9, 0.9]),
        ("Single curved", [0.1, 0.9, 0.5, 0.1, 0.9, 0.9]),
        ("X shape (2 splines)", [0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.1, 0.9, 0.5, 0.5, 0.9, 0.1]),
        ("Triangle (3 splines)", [0.1, 0.8, 0.5, 0.1, 0.9, 0.8, 0.1, 0.8, 0.1, 0.5, 0.5, 0.1, 0.9, 0.8, 0.9, 0.5, 0.5, 0.1]),
        (
            "Square (4 splines)",
            [0.1, 0.1, 0.5, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.5, 0.9, 0.9, 0.9, 0.9, 0.5, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.5, 0.1, 0.1],
        ),
        ("Random (seed=42)", np.random.default_rng(42).uniform(0, 1, 18).tolist()),
    ]

    labels = [t[0] for t in test_cases]
    raw_params = [jnp.array(t[1]) for t in test_cases]

    # Pad shorter arrays to the same length so we can stack into a batch
    max_len = max(p.shape[0] for p in raw_params)
    padded = []
    for p in raw_params:
        if p.shape[0] < max_len:
            reps = (max_len - p.shape[0]) // 6
            p = jnp.concatenate([p, jnp.tile(p[-6:], reps)])
        padded.append(p)

    batch = jnp.stack(padded)  # (N, max_len)
    canvases = paint_multiple_splines(batch, IMAGE_DIM)
    print(f"Input shape:  {batch.shape}")
    print(f"Output shape: {canvases.shape}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, label, canvas in zip(axes.flatten(), labels, canvases):
        ax.imshow(np.array(canvas), cmap="gray_r", vmin=0, vmax=1, origin="upper")
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.suptitle("paint_multiple_splines — test renders", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("test_renders.png", dpi=150)
    print("Saved test_renders.png")
    plt.show()
