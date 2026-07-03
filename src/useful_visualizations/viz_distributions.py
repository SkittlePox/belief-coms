"""Interactive view of how ``JointCategoricalPair`` models a joint over two categoricals
as a single flat ``distrax.Categorical``.

Edit the joint ``P(var1, var2)`` grid in the control panel and everything recomputes
live in the browser -- the flat vector, both marginals, a conditional (pick which
``var1`` to condition on), and an empirical grid from fresh samples. The initial values
reproduce ``distributions.py``'s worked example (mass split over flat indices 1 and 2).

``JointCategoricalPair``'s operations are trivial linear algebra (reshape, sum an axis,
row-normalize, inverse-CDF sample), so they are reimplemented in a small post-render
script to give live updates without a server; the Python side just lays out the panels.
"""

from __future__ import annotations

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from plotly.subplots import make_subplots

from tools.distributions import JointCategoricalPair

from . import _figures as F

V1, V2 = 3, 5  # var1 has 3 categories, var2 has 5 -> flat length 15


def _example():
    """The distributions.py __main__ example: mass split over flat indices 1 and 2."""
    factory = JointCategoricalPair(vars_num_categories=(V1, V2))
    probs = jnp.zeros(V1 * V2).at[1:3].set(jnp.array([0.5, 0.5]))
    return factory, distrax.Categorical(probs=probs)


# Live editor: reimplements the JointCategoricalPair ops in JS and drives the six panels
# with Plotly.restyle. Trace order is fixed by the add order in render():
#   0 flat vector | 1 joint grid | 2 P(var1) | 3 P(var2) | 4 conditional | 5 empirical
_INTERACTIVE_JS = """
var gd = document.getElementById('{plot_id}');
(function () {
  function ready() {
    // Let the page scroll under a fixed-height plot so the controls sit above it.
    document.documentElement.style.height = 'auto';
    document.body.style.height = 'auto';
    document.body.style.margin = '0';
    gd.style.height = '820px';

    var panel = document.createElement('div');
    panel.style.cssText = 'font-family:Inter,Helvetica,Arial,sans-serif;padding:14px 20px;'
      + 'background:#eef3f7;border-bottom:1px solid #d8e0e6;';
    var head = document.createElement('div');
    head.textContent = 'Edit the joint  P(var1, var2)  \\u2014  everything below updates live';
    head.style.cssText = 'font-weight:600;margin-bottom:8px;color:#1f2933;';
    panel.appendChild(head);

    var tbl = document.createElement('table');
    tbl.style.borderCollapse = 'collapse';
    var hr = document.createElement('tr');
    hr.appendChild(document.createElement('th'));
    for (var j = 0; j < 5; j++) {
      var th = document.createElement('th');
      th.textContent = 'v2=' + j;
      th.style.cssText = 'padding:2px 6px;font-weight:500;color:#7b8794;';
      hr.appendChild(th);
    }
    tbl.appendChild(hr);
    var inp = [];
    for (var i = 0; i < 3; i++) {
      var tr = document.createElement('tr');
      var rl = document.createElement('td');
      rl.textContent = 'v1=' + i;
      rl.style.cssText = 'padding:2px 8px;color:#7b8794;';
      tr.appendChild(rl);
      inp.push([]);
      for (var j = 0; j < 5; j++) {
        var td = document.createElement('td');
        var box = document.createElement('input');
        box.type = 'number'; box.step = '0.05'; box.min = '0'; box.value = '0';
        box.style.cssText = 'width:58px;padding:5px;margin:2px;border:1px solid #c3ccd4;'
          + 'border-radius:4px;text-align:center;';
        box.addEventListener('input', update);
        td.appendChild(box); tr.appendChild(td); inp[i].push(box);
      }
      tbl.appendChild(tr);
    }
    panel.appendChild(tbl);

    var ctr = document.createElement('div');
    ctr.style.cssText = 'margin-top:10px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;';
    var condLbl = document.createElement('label');
    condLbl.textContent = 'condition on  var1 = '; condLbl.style.color = '#1f2933';
    var sel = document.createElement('select');
    for (var i = 0; i < 3; i++) {
      var o = document.createElement('option'); o.value = i; o.text = i; sel.appendChild(o);
    }
    sel.style.cssText = 'padding:4px;border:1px solid #c3ccd4;border-radius:4px;';
    sel.addEventListener('change', update); condLbl.appendChild(sel);
    var sampLbl = document.createElement('label');
    sampLbl.textContent = 'samples: '; sampLbl.style.color = '#1f2933';
    var samp = document.createElement('input');
    samp.type = 'number'; samp.min = '100'; samp.step = '500'; samp.value = '3000';
    samp.style.cssText = 'width:84px;padding:4px;border:1px solid #c3ccd4;border-radius:4px;';
    samp.addEventListener('input', update); sampLbl.appendChild(samp);
    function mkBtn(t) {
      var b = document.createElement('button'); b.textContent = t;
      b.style.cssText = 'padding:6px 14px;border:none;border-radius:6px;background:#2c7fb8;'
        + 'color:#fff;cursor:pointer;font-weight:600;';
      return b;
    }
    var bReset = mkBtn('reset example'), bUnif = mkBtn('uniform'), bClear = mkBtn('clear');
    ctr.appendChild(condLbl); ctr.appendChild(sampLbl);
    ctr.appendChild(bReset); ctr.appendChild(bUnif); ctr.appendChild(bClear);
    panel.appendChild(ctr);
    var note = document.createElement('div');
    note.style.cssText = 'margin-top:6px;font-size:12px;color:#7b8794;';
    panel.appendChild(note);
    // Plotly wraps the graph div in a container, so gd is not a child of <body>.
    // Insert the panel just before whichever element actually sits in the flow.
    var container = gd.parentNode;
    if (container === document.body) {
      document.body.insertBefore(panel, gd);
    } else {
      container.parentNode.insertBefore(panel, container);
    }

    function setGrid(v) {
      for (var i = 0; i < 3; i++) for (var j = 0; j < 5; j++) inp[i][j].value = v[i][j];
    }
    bReset.onclick = function () {
      setGrid([[0, 0.5, 0.5, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]); update();
    };
    bUnif.onclick = function () {
      var g = []; for (var i = 0; i < 3; i++) { g.push([]); for (var j = 0; j < 5; j++) g[i].push((1 / 15).toFixed(4)); }
      setGrid(g); update();
    };
    bClear.onclick = function () {
      var g = []; for (var i = 0; i < 3; i++) { g.push([]); for (var j = 0; j < 5; j++) g[i].push(0); }
      setGrid(g); update();
    };

    function readGrid() {
      var g = [];
      for (var i = 0; i < 3; i++) { g.push([]); for (var j = 0; j < 5; j++) {
        var v = parseFloat(inp[i][j].value); g[i].push(isNaN(v) || v < 0 ? 0 : v);
      } }
      return g;
    }
    function sum2(g) { var s = 0; for (var i = 0; i < 3; i++) for (var j = 0; j < 5; j++) s += g[i][j]; return s; }
    function flat(g) { var f = []; for (var i = 0; i < 3; i++) for (var j = 0; j < 5; j++) f.push(g[i][j]); return f; }
    function maxOf(g) { var m = 0; for (var i = 0; i < 3; i++) for (var j = 0; j < 5; j++) if (g[i][j] > m) m = g[i][j]; return m || 1; }
    function margV1(g) { return g.map(function (r) { return r.reduce(function (a, b) { return a + b; }, 0); }); }
    function margV2(g) { var m = [0, 0, 0, 0, 0]; for (var i = 0; i < 3; i++) for (var j = 0; j < 5; j++) m[j] += g[i][j]; return m; }
    function condV2(g, k) {
      var row = g[k], s = row.reduce(function (a, b) { return a + b; }, 0);
      return s > 0 ? row.map(function (v) { return v / s; }) : [0, 0, 0, 0, 0];
    }
    function empirical(g, N) {
      var f = flat(g), cum = [], c = 0;
      for (var x = 0; x < f.length; x++) { c += f[x]; cum.push(c); }
      var cnt = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]];
      for (var n = 0; n < N; n++) {
        var r = Math.random(), idx = 0;
        while (idx < 14 && r > cum[idx]) idx++;
        cnt[Math.floor(idx / 5)][idx % 5]++;
      }
      for (var i = 0; i < 3; i++) for (var j = 0; j < 5; j++) cnt[i][j] /= N;
      return cnt;
    }

    function update() {
      var raw = readGrid(), s = sum2(raw);
      var g = raw;
      if (s > 0) { g = raw.map(function (r) { return r.map(function (v) { return v / s; }); }); }
      var k = parseInt(sel.value), N = Math.max(1, parseInt(samp.value) || 3000);
      var gm = maxOf(g), emp = empirical(g, N), em = maxOf(emp);
      Plotly.restyle(gd, { z: [[flat(g)]], zmax: [gm] }, [0]);
      Plotly.restyle(gd, { z: [g], zmax: [gm] }, [1]);
      Plotly.restyle(gd, { y: [margV1(g)] }, [2]);
      Plotly.restyle(gd, { y: [margV2(g)] }, [3]);
      Plotly.restyle(gd, { y: [condV2(g, k)] }, [4]);
      Plotly.restyle(gd, { z: [emp], zmax: [em] }, [5]);
      var anns = gd.layout.annotations || [];
      for (var a = 0; a < anns.length; a++) {
        if (anns[a].text && anns[a].text.indexOf('conditional') === 0) {
          var o = {}; o['annotations[' + a + '].text'] = 'conditional  P(var2 | var1=' + k + ')';
          Plotly.relayout(gd, o); break;
        }
      }
      note.textContent = s > 0
        ? ('input sum = ' + s.toFixed(3) + '  \\u2192  normalized to 1')
        : 'all zero \\u2014 enter some probability mass';
      Plotly.Plots.resize(gd);
    }

    bReset.onclick();  // paint the example on load
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready);
  } else { ready(); }
})();
"""


def render() -> str:
    factory, dist = _example()
    probs = np.asarray(dist.probs)
    grid = probs.reshape(V1, V2)

    marg_v1 = np.asarray(factory.marginalize_var2(dist).probs)  # sum out var2 -> P(var1)
    marg_v2 = np.asarray(factory.marginalize_var1(dist).probs)  # sum out var1 -> P(var2)
    cond_v2_given_v1_0 = np.asarray(factory.conditional_var2_given_var1(dist, 0).probs)

    keys = jax.random.split(jax.random.key(0), 3000)
    samples = np.asarray(jax.vmap(lambda k: factory.sample_joint_distribution(k, dist))(keys))
    counts = np.zeros((V1, V2))
    for v1, v2 in samples:
        counts[v1, v2] += 1
    emp = counts / counts.sum()

    fig = make_subplots(
        rows=3, cols=3,
        specs=[
            [{"colspan": 3}, None, None],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.22, 0.39, 0.39],
        vertical_spacing=0.13, horizontal_spacing=0.09,
        subplot_titles=(
            "flat distrax.Categorical  —  probs[i],  i = var1·5 + var2   (row-major)",
            "joint grid  P(var1, var2)  =  probs.reshape(3, 5)",
            "marginal  P(var1)  =  marginalize_var2",
            "marginal  P(var2)  =  marginalize_var1",
            "conditional  P(var2 | var1=0)",
            "empirical grid  (from samples)",
            "",
        ),
    )

    # Trace 0: flat vector (1-row heatmap over the 15 flat indices).
    fig.add_trace(
        F.heatmap_trace(
            probs[None, :], x=[str(i) for i in range(V1 * V2)], y=["p"],
            colorscale=F.PROB_SCALE, zmin=0.0, zmax=probs.max(), showscale=False,
            hover="prob", dynamic_text=True,
        ),
        row=1, col=1,
    )
    # Trace 1: the joint grid.
    fig.add_trace(
        F.heatmap_trace(
            grid, x=[f"v2={j}" for j in range(V2)], y=[f"v1={i}" for i in range(V1)],
            colorscale=F.PROB_SCALE, zmin=0.0, zmax=grid.max(), showscale=False,
            hover="prob", dynamic_text=True,
        ),
        row=2, col=1,
    )
    # Traces 2 & 3: the two marginals.
    fig.add_trace(F.bar_trace(marg_v1, [f"v1={i}" for i in range(V1)], dynamic_text=True), row=2, col=2)
    fig.add_trace(F.bar_trace(marg_v2, [f"v2={j}" for j in range(V2)], dynamic_text=True), row=2, col=3)

    # Trace 4: conditional. Trace 5: empirical grid.
    fig.add_trace(
        F.bar_trace(cond_v2_given_v1_0, [f"v2={j}" for j in range(V2)], color=F.ACCENT_2, dynamic_text=True),
        row=3, col=1,
    )
    fig.add_trace(
        F.heatmap_trace(
            emp, x=[f"v2={j}" for j in range(V2)], y=[f"v1={i}" for i in range(V1)],
            colorscale=F.PROB_SCALE, zmin=0.0, zmax=grid.max(), showscale=False,
            hover="freq", dynamic_text=True,
        ),
        row=3, col=2,
    )

    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=2)
    for r, c in [(2, 2), (2, 3), (3, 1)]:
        fig.update_yaxes(range=[0, 1.15], row=r, col=c)

    fig.update_layout(
        height=820,
        margin=dict(l=60, r=30, t=70, b=50),
        title="JointCategoricalPair — one flat categorical models a joint over two variables",
    )
    return F.write(fig, "viz_distributions", post_script=_INTERACTIVE_JS)


if __name__ == "__main__":
    print(render())
