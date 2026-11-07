"""Draw the stack geometry, heater configuration and operating protocol."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.append(str(ROOT / 'research' / 'deps'))
OUT = ROOT / 'regenerated_figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8.3,
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 160, 'savefig.dpi': 600,
    'axes.unicode_minus': True, 'lines.linewidth': .9,
})
C = {'ink': '#171717', 'line': '#3A3A3A', 'ep': '#F4F4F4',
     'bp': '#FFFFFF', 'gdl': '#FFFFFF', 'cl': '#FFFFFF',
     'pem': '#DFDFDF', 'heat': '#242424', 'load': '#242424',
     'window': '#EDEDED', 'mesh': '#909090'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources():
    source = ROOT / '代码' / '模型物理.py'
    spec = importlib.util.spec_from_file_location('fig1_physics_check', source)
    physics = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = physics
    spec.loader.exec_module(physics)
    m = physics.build_shared(n_cells=5, refine=1, endplates=True)
    assert len(m.mea_indices) == 5
    kinds = ('AGDL', 'ACL', 'PEM', 'CCL', 'CGDL')
    widths = [float(m.dx[(m.cell_ids == 0) & (m.kind == k)].sum()*1e6) for k in kinds]
    assert np.allclose(widths, [150, 3.4, 12, 11.3, 150])
    assert np.isclose(m.dx.sum()*1e3, 33.6335)
    bp_ownership = []
    for i, ids in enumerate(m.bp_indices):
        n = len(ids)//2
        sides = [float(m.dx[ids[:n]].sum()*1e3), float(m.dx[ids[n:]].sum()*1e3)]
        assert np.allclose(sides, [2 if i == 0 else 1, 2 if i == 4 else 1])
        test = np.zeros(5); test[i] = 25.
        distributed = m.source_from_cell_power(test, where='BP')
        assert np.isclose(distributed.sum(), 25.)
        actual = [float(distributed[ids[:n]].sum()), float(distributed[ids[n:]].sum())]
        assert np.allclose(np.array(actual)/25., np.array(sides)/sum(sides))
        bp_ownership.append({'cell': i+1, 'left_mm': sides[0], 'right_mm': sides[1],
                             'left_power_fraction': actual[0]/25., 'right_power_fraction': actual[1]/25.})
    protocol_path = ROOT / 'research' / 'study_v2' / 'protocol.json'
    p = json.loads(protocol_path.read_text(encoding='utf-8-sig'))
    op = p['operating_protocol']
    assert p['model']['cells'] == 5 and op['horizon_s'] == 180
    assert p['model']['active_area_cm2'] == 25
    assert op['window_s'] == [120, 180]
    assert op['temperature_reserves_C'] == [.01, 1, 3]
    assert op['current_A_cm2'] == 'min(0.005*max(t,0),0.30)'
    assert p['design_families']['uniform']['bounds'][1] == [10, 90]
    return {'layer_order': list(kinds), 'layer_thickness_um': widths,
            'physical_BPs': 6, 'endplates': 2, 'MEA_count': 5,
            'stack_thickness_mm': float(m.dx.sum()*1e3), 'BP_ownership': bp_ownership,
            'sources': [{'path': str(source.relative_to(ROOT)), 'sha256': digest(source)},
                        {'path': str(protocol_path.relative_to(ROOT)), 'sha256': digest(protocol_path)}]}


def rect(ax, x, y, w, h, color, lw=.6, zorder=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=C['line'],
                           linewidth=lw, zorder=zorder))


def text(ax, x, y, s, size=8.3, **kw):
    return ax.text(x, y, s, fontsize=size, color=kw.pop('color', C['ink']), ha=kw.pop('ha', 'center'),
                   va=kw.pop('va', 'center'), **kw)


def line(ax, x, y, **kw):
    ax.plot(x, y, color=kw.pop('color', C['line']), lw=kw.pop('lw', .65), **kw)


def arrow(ax, xy, xytext, color=None, style='-|>', lw=.85):
    ax.annotate('', xy=xy, xytext=xytext,
                arrowprops={'arrowstyle': style, 'lw': lw, 'color': color or C['line'],
                            'shrinkA': 0, 'shrinkB': 0, 'mutation_scale': 6.5})


def title(ax, label, s):
    text(ax, 0, 96, label, size=10, fontweight='bold', ha='left')
    text(ax, 4.5, 96, s, size=9.1, ha='left')


def panel_a(ax):
    title(ax, 'a', 'Stack geometry and heater allocation')
    text(ax, 99, 96, r'$A$ = 25 cm²', size=8.3, ha='right')
    x0, epw, bpw, meaw = 3.5, 7., 3.8, 11.24
    # Schematic overview: MEAs enlarged relative to the plates for legibility.
    y, h = 43, 24
    x = x0
    rect(ax, x, y, epw, h, C['ep']); text(ax, x+epw/2, y+h/2, 'EP', size=8)
    x += epw
    bps, centers = [], []
    for n in range(6):
        bps.append(x)
        rect(ax, x, y, bpw, h, C['bp'])
        text(ax, x+bpw/2, 77, 'BP', size=8)
        if n not in (0, 5):
            line(ax, [x+bpw/2]*2, [y, y+h], color=C['line'], lw=.5, ls=(0, (2, 2)))
        x += bpw
        if n < 5:
            centers.append(x+meaw/2)
            rect(ax, x, y, meaw, h, C['gdl'])
            # Thin electrode/membrane trio remains contiguous in this overview.
            for offset, width, color in [(meaw*.41, meaw*.035, C['cl']),
                                          (meaw*.445, meaw*.11, C['pem']),
                                          (meaw*.555, meaw*.045, C['cl'])]:
                rect(ax, x+offset, y, width, h, color, lw=.25)
            text(ax, x+meaw/2, 77, f'Cell {n+1}', size=8)
            text(ax, x+meaw/2, 34, 'MEA', size=8)
            x += meaw
    rect(ax, x, y, epw, h, C['ep']); text(ax, x+epw/2, y+h/2, 'EP', size=8)
    # The source term is distributed in the cell-owned BP regions; the arrows
    # identify those regions, not equal power splitting at the end cells.
    controls = [r'$q_{\mathrm{e}}$', r'$q_{\mathrm{i}}$', r'$q_{\mathrm{c}}$',
                r'$q_{\mathrm{i}}$', r'$q_{\mathrm{e}}$']
    for n, xc in enumerate(centers):
        target_left = bps[n]+bpw*(.5 if n == 0 else .75)
        target_right = bps[n+1]+bpw*(.5 if n == 4 else .25)
        for target in (target_left, target_right):
            arrow(ax, (target, y-1.5), (xc, 20), color=C['heat'])
        text(ax, xc, 10, controls[n], size=10.3, color=C['heat'])


def dimension(ax, start, end, y, label, label_y=None):
    arrow(ax, (end, y), (start, y), style='<->', lw=.65)
    for x in (start, end):
        line(ax, [x, x], [y-2.3, y+2.3], lw=.55)
    if label:
        text(ax, (start+end)/2, y-8 if label_y is None else label_y, label, size=8)


def panel_b(ax):
    title(ax, 'b', 'MEA layer structure')
    # Zoom at the top and full MEA beneath, both proportional to actual thickness.
    lower_x, lower_w = 7., 86.
    zoom_x, zoom_w = 28., 57.
    full = [150., 3.4, 12., 11.3, 150.]
    cols = [C['gdl'], C['cl'], C['pem'], C['cl'], C['gdl']]
    cumulative = np.r_[0, np.cumsum(full)]
    edge = lower_x+lower_w*cumulative/sum(full)
    y0, hh = 25., 18.
    for i in range(5):
        rect(ax, edge[i], y0, edge[i+1]-edge[i], hh, cols[i])
        # Four visible GDL volumes, following the original base spatial mesh.
        if i in (0, 4):
            for frac in (.25, .5, .75):
                xx = edge[i]+frac*(edge[i+1]-edge[i])
                line(ax, [xx, xx], [y0, y0+hh], color=C['mesh'], lw=.35)
    text(ax, (edge[0]+edge[1])/2, 17., 'Anode GDL (150 µm)', size=8)
    text(ax, (edge[4]+edge[5])/2, 17., 'Cathode GDL (150 µm)', size=8)
    zedge = zoom_x+zoom_w*np.r_[0, np.cumsum(full[1:4])]/sum(full[1:4])
    for i, (name, width, color, divisions) in enumerate(zip(
            ['ACL', 'PEM', 'CCL'], full[1:4], cols[1:4], [2, 4, 2])):
        rect(ax, zedge[i], 59, zedge[i+1]-zedge[i], 13., color)
        for frac in np.arange(1, divisions)/divisions:
            xx = zedge[i]+frac*(zedge[i+1]-zedge[i])
            line(ax, [xx, xx], [59, 72], color=C['mesh'], lw=.35)
        text(ax, (zedge[i]+zedge[i+1])/2, 87, name, size=8)
        text(ax, (zedge[i]+zedge[i+1])/2, 77, f'{width:g} µm', size=8)
    # Dashed guides show the magnified central zone. These are not material gaps.
    line(ax, [edge[1], zedge[0]], [y0+hh+1, 58], ls=(0, (3, 3)), lw=.65)
    line(ax, [edge[4], zedge[-1]], [y0+hh+1, 58], ls=(0, (3, 3)), lw=.65)
    dimension(ax, lower_x, lower_x+lower_w, 5, '326.7 µm', label_y=-5)


def panel_c(ax):
    title(ax, 'c', 'Cold-start operating protocol')
    x = lambda t: 7.+86.*t/180.
    tau = 48.  # Illustration only, not an optimized or measured cutoff.
    y_hi, y_off = 73., 59.
    line(ax, [x(0), x(tau), x(tau), x(180)], [y_hi, y_hi, y_off, y_off], color=C['heat'], lw=.9)
    text(ax, x(24), 83, 'Heater on', size=8)
    text(ax, x(125), 68, 'Heater off', size=8)
    text(ax, x(tau), 83, 'τ', size=9.4)
    text(ax, x(93), 83, '10 ≤ τ ≤ 90 s', size=8)
    line(ax, [x(0), x(60), x(180)], [33, 46, 46], color=C['load'], lw=.9)
    text(ax, x(26), 46, 'Current ramp', size=8)
    text(ax, x(130), 52, '0.30 A cm⁻²', size=8)
    rect(ax, x(120), 18, x(180)-x(120), 14, C['window'], lw=.65)
    text(ax, x(150), 25, r'All cell means ≥ $T_{\mathrm{req}}$', size=8)
    text(ax, x(0), 23, r'$T_{\mathrm{req}}$ = 0.01, 1 or 3 °C', size=8, ha='left')
    line(ax, [x(0), x(180)], [10, 10], color=C['ink'])
    for t in [0, 30, 60, 90, 120, 150, 180]:
        line(ax, [x(t)]*2, [10, 7.3], color=C['ink'])
        text(ax, x(t), 0.7, str(t), size=8)
    text(ax, x(90), -10, 'Time (s)', size=8.3)


def main():
    source_audit = check_sources()
    fig = plt.figure(figsize=(6.6929133858, 4.6456692913), facecolor='white')
    gs = fig.add_gridspec(3, 1, left=.025, right=.99, bottom=.06, top=.99,
                          height_ratios=[.85, 1.18, 1.05], hspace=.055)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax in axes:
        ax.set(xlim=(0, 100), ylim=(-12, 101))
        ax.axis('off')
    axes[0].set_ylim(0, 101)
    axes[2].set_ylim(-18, 101)
    panel_a(axes[0]); panel_b(axes[1]); panel_c(axes[2])
    fig.canvas.draw()
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(OUT/f'Fig1.{suffix}', dpi=600, facecolor='white')

    plt.close(fig)

if __name__ == '__main__': main()
