"""Render every visualization in one shot.

    uv run python -m useful_visualizations
"""

from __future__ import annotations

from . import viz_distributions, viz_guessing_game, viz_stacked


def main() -> None:
    renderers = [
        ("distributions", viz_distributions.render),
        ("guessing game", viz_guessing_game.render),
        ("stacked signification", viz_stacked.render),
    ]
    print("Rendering useful visualizations:")
    for name, render in renderers:
        path = render()
        print(f"  ✓ {name:<24} → {path}")


if __name__ == "__main__":
    main()
