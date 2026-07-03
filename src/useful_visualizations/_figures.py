"""Shared Plotly vocabulary for the useful_visualizations package.

Every view in this package draws from the small set of builders here so the whole
suite reads as one system: one template, one probability colorscale, one diverging
reward scale, one accent, and one temporal-slider driver. Keep this module tiny --
if a view needs a new visual idiom, add it here (once) rather than styling inline.

Two levels are exposed:
  * trace builders (``bar_trace`` / ``heatmap_trace``) return bare go traces so the
    dashboards can drop them into ``make_subplots`` cells and animate them, and
  * figure builders (``prob_bars`` / ``heatmap`` / ``belief_matrix``) wrap a single
    trace into a themed standalone ``go.Figure`` for the static sections.

``attach_slider`` turns a base figure + a list of ``go.Frame``s into a scrubbable
animation (used by the belief-by-waiting rollout and the step_env trace). ``write``
serializes any figure to a self-contained HTML file under ``renders/``.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


# --------------------------------------------------------------------------- #
# One template + palette for the whole suite (the harmony layer).
# --------------------------------------------------------------------------- #
INK = "#1f2933"        # primary text
MUTED = "#7b8794"      # secondary text / gridlines
ACCENT = "#2c7fb8"     # bars (a calm blue)
ACCENT_2 = "#d95f0e"   # a warm second accent (highlights: acts / boundaries)
PANEL_BG = "#ffffff"
FIG_BG = "#f7f9fb"

# Sequential scale for probabilities (white -> teal). Reads as "mass here".
PROB_SCALE = [
    [0.0, "#f7fcfd"],
    [0.25, "#ccece6"],
    [0.5, "#66c2a4"],
    [0.75, "#238b8b"],
    [1.0, "#00441b"],
]
# Diverging scale for rewards (red -> neutral -> green).
REWARD_SCALE = "RdYlGn"

_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color=INK),
        paper_bgcolor=FIG_BG,
        plot_bgcolor=PANEL_BG,
        colorway=[ACCENT, ACCENT_2, "#66c2a4", "#7b8794", "#8856a7"],
        margin=dict(l=60, r=30, t=60, b=50),
        title=dict(x=0.5, xanchor="center", font=dict(size=18)),
        xaxis=dict(gridcolor="#eceff1", zerolinecolor="#e0e4e8", linecolor=MUTED),
        yaxis=dict(gridcolor="#eceff1", zerolinecolor="#e0e4e8", linecolor=MUTED),
    )
)
pio.templates["belief_coms"] = _TEMPLATE
pio.templates.default = "belief_coms"


# --------------------------------------------------------------------------- #
# Trace builders (bare traces, for subplots + animation frames).
# --------------------------------------------------------------------------- #
def bar_trace(values, labels=None, name=None, color=ACCENT, showlegend=False, dynamic_text=False):
    """A categorical distribution / vector as a bar trace, value labels on top.

    ``dynamic_text=True`` makes the labels follow ``y`` (``texttemplate``) instead of a
    baked-in text list, so a later ``Plotly.restyle`` of ``y`` updates the labels too.
    """
    values = np.asarray(values, dtype=float)
    if labels is None:
        labels = [str(i) for i in range(len(values))]
    labels = [str(l) for l in labels]
    kwargs = dict(
        x=labels, y=values, name=name, marker_color=color,
        textposition="outside", cliponaxis=False, showlegend=showlegend,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    )
    if dynamic_text:
        kwargs["texttemplate"] = "%{y:.2f}"
    else:
        kwargs["text"] = [f"{v:.2f}" for v in values]
    return go.Bar(**kwargs)


def heatmap_trace(
    matrix,
    x=None,
    y=None,
    colorscale=None,
    zmin=None,
    zmax=None,
    text=True,
    text_fmt="{:.2f}",
    coloraxis=None,
    showscale=True,
    hover="value",
    dynamic_text=False,
):
    """A 2D array as a heatmap trace with optional per-cell value annotations.

    ``y`` is reversed onto the axis so row 0 sits at the top (matrix reading order).
    Pass ``coloraxis='coloraxis'`` to share one colorbar across subplots.
    ``dynamic_text=True`` makes cell labels follow ``z`` (``%{z:.2f}``) so a later
    ``Plotly.restyle`` of ``z`` updates the labels too.
    """
    matrix = np.asarray(matrix, dtype=float)
    n_rows, n_cols = matrix.shape
    if x is None:
        x = [str(i) for i in range(n_cols)]
    if y is None:
        y = [str(i) for i in range(n_rows)]
    x = [str(v) for v in x]
    y = [str(v) for v in y]

    text_arr = None
    texttemplate = None
    if dynamic_text:
        texttemplate = "%{z:.2f}"
    elif text:
        text_arr = [[text_fmt.format(v) for v in row] for row in matrix]
        texttemplate = "%{text}"

    kwargs = dict(
        z=matrix,
        x=x,
        y=y,
        text=text_arr,
        texttemplate=texttemplate,
        textfont=dict(size=11),
        hovertemplate=f"row %{{y}}, col %{{x}}<br>{hover}=%{{z:.3f}}<extra></extra>",
    )
    if coloraxis is not None:
        kwargs["coloraxis"] = coloraxis
    else:
        kwargs.update(colorscale=colorscale or PROB_SCALE, zmin=zmin, zmax=zmax, showscale=showscale)
    trace = go.Heatmap(**kwargs)
    # Put row 0 at the top.
    return trace


# --------------------------------------------------------------------------- #
# Figure builders (standalone themed figures for static sections).
# --------------------------------------------------------------------------- #
def prob_bars(probs, labels=None, title="", x_title="", y_title="probability", color=ACCENT):
    fig = go.Figure(bar_trace(probs, labels, color=color))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    fig.update_yaxes(range=[0, 1.05])
    return fig


def heatmap(
    matrix, x=None, y=None, title="", x_title="", y_title="",
    colorscale=None, zmin=None, zmax=None, text=True, text_fmt="{:.2f}", reversed_y=True,
):
    fig = go.Figure(heatmap_trace(matrix, x, y, colorscale, zmin, zmax, text, text_fmt))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    if reversed_y:
        fig.update_yaxes(autorange="reversed")
    return fig


def belief_matrix(probs_2d, row_labels=None, col_labels=None, title="", y_title="agent"):
    """A stack of categorical distributions ``[rows, S]`` as a probability heatmap."""
    probs_2d = np.asarray(probs_2d, dtype=float)
    fig = heatmap(
        probs_2d, x=col_labels, y=row_labels, title=title,
        x_title="state", y_title=y_title, colorscale=PROB_SCALE, zmin=0.0, zmax=1.0,
    )
    return fig


# --------------------------------------------------------------------------- #
# Temporal driver: base figure + frames -> scrubbable animation.
# --------------------------------------------------------------------------- #
def attach_slider(fig: go.Figure, frames: Sequence[go.Frame], labels: Sequence[str],
                  slider_prefix: str = "") -> go.Figure:
    """Attach a step slider to ``fig`` over ``frames`` (no play/pause).

    Each frame's ``name`` is used as its slider target; ``labels[i]`` is the slider
    tick text. Frames may carry ``layout`` updates (e.g. a per-step title) which are
    redrawn on scrub. The base figure should already show frame 0. ``write`` injects
    the "◂ Back / Step ▸" buttons that advance the slider one frame at a time.
    """
    fig.frames = list(frames)

    steps = [
        dict(
            method="animate",
            label=labels[i],
            args=[[frames[i].name], dict(mode="immediate",
                                         frame=dict(duration=0, redraw=True),
                                         transition=dict(duration=0))],
        )
        for i in range(len(frames))
    ]
    slider = dict(
        active=0, x=0.0, y=0.0, len=1.0, pad=dict(t=40, b=10),
        currentvalue=dict(prefix=slider_prefix, font=dict(size=13, color=MUTED)),
        steps=steps,
    )
    fig.update_layout(sliders=[slider])
    return fig


# Injected after render (via write's post_script) for any figure that has frames:
# a "◂ Back / Step ▸" control that advances the existing slider exactly one frame.
# Plotly's declarative buttons can only play *all* frames, so single-stepping needs
# this. It drives the slider's own steps (no private APIs) and keeps the handle synced.
_STEP_BUTTONS_JS = """
var gd = document.getElementById('{plot_id}');
(function () {
  function setup() {
    var sliders = gd.layout && gd.layout.sliders;
    if (!sliders || !sliders.length || !sliders[0].steps || !sliders[0].steps.length) return;
    var steps = sliders[0].steps;
    var n = steps.length;
    function frameName(i) { return steps[i].args[0][0]; }
    function current() {
      var a = gd.layout.sliders[0].active;
      return (typeof a === 'number') ? a : 0;
    }
    function goTo(i) {
      i = ((i % n) + n) % n;
      Plotly.animate(gd, [frameName(i)],
        {mode: 'immediate', frame: {duration: 0, redraw: true}, transition: {duration: 0}});
      Plotly.relayout(gd, {'sliders[0].active': i});
    }
    // Float the controls as a fixed overlay (bottom-right, near the slider) so they
    // don't consume layout height and push the plot off a full-viewport page.
    var wrap = document.createElement('div');
    wrap.style.cssText =
      'position:fixed;bottom:16px;right:24px;z-index:1000;' +
      'font-family:Inter,Helvetica,Arial,sans-serif;';
    function mkBtn(label) {
      var b = document.createElement('button');
      b.textContent = label;
      b.style.cssText =
        'margin:0 5px;padding:8px 18px;font-size:14px;font-weight:600;border:none;' +
        'border-radius:7px;background:#2c7fb8;color:#fff;cursor:pointer;' +
        'box-shadow:0 1px 4px rgba(31,41,51,0.25);';
      b.onmouseenter = function () { b.style.background = '#1f6690'; };
      b.onmouseleave = function () { b.style.background = '#2c7fb8'; };
      return b;
    }
    var back = mkBtn('\\u25C2 Back');
    var step = mkBtn('Step \\u25B8');
    back.onclick = function () { goTo(current() - 1); };
    step.onclick = function () { goTo(current() + 1); };
    wrap.appendChild(back);
    wrap.appendChild(step);
    document.body.appendChild(wrap);
  }
  if (gd) { setup(); }
})();
"""


# --------------------------------------------------------------------------- #
# Output.
# --------------------------------------------------------------------------- #
def renders_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), "renders")
    os.makedirs(d, exist_ok=True)
    return d


def write(fig: go.Figure, name: str, post_script=None) -> str:
    """Write ``fig`` to ``renders/<name>.html`` (self-contained) and return the path.

    Figures that carry frames get the "◂ Back / Step ▸" buttons injected via a
    post-render script (``{plot_id}`` is substituted with the div id by Plotly).
    ``post_script`` (str or list) adds further per-page scripts, e.g. the interactive
    editor in ``viz_distributions``.
    """
    path = os.path.join(renders_dir(), f"{name}.html")
    scripts = []
    if fig.frames:
        scripts.append(_STEP_BUTTONS_JS)
    if post_script:
        scripts.extend(post_script if isinstance(post_script, (list, tuple)) else [post_script])
    post = scripts or None
    # Fill the browser width (widescreen-friendly) instead of a fixed pixel box; the
    # module sets a comfortable height. autosize + default_width="100%" + a responsive
    # config makes the figure track the window on resize.
    fig.update_layout(autosize=True)
    fig.layout.width = None
    # Width flexes to the window; height is pinned to the figure's pixel height so it
    # doesn't collapse in a full-page body with no explicit height.
    default_height = f"{int(fig.layout.height)}px" if fig.layout.height else "800px"
    fig.write_html(
        path, include_plotlyjs="inline", full_html=True, post_script=post,
        default_width="100%", default_height=default_height, config={"responsive": True},
    )
    return path
