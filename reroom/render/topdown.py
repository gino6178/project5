"""Top-down floor-plan rendering -- the paper's main figure format.

Experiment two of the plan ("floor geometry difficulty") is meant to be one of
the paper's headline figures, because a top-down comparison is exactly what
shows the difference between *reconstruction* and *retargeting*.
"""
from __future__ import annotations

import math
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from ..core.categories import CORE_CATEGORIES, prior
from ..core.scene import Scene
from ..geom.polygon import as_polygon

__all__ = ["draw_scene", "figure_comparison", "CATEGORY_COLORS"]

_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#8C6BB1", "#7FC97F",
]
CATEGORY_COLORS: dict[str, str] = {}


def _color(cat: str) -> str:
    if cat not in CATEGORY_COLORS:
        CATEGORY_COLORS[cat] = _PALETTE[len(CATEGORY_COLORS) % len(_PALETTE)]
    return CATEGORY_COLORS[cat]


def draw_scene(ax, scene: Scene, title: str | None = None,
               labels: bool = True, show_openings: bool = True,
               show_front: bool = True, alpha: float = 0.85,
               highlight: Sequence[str] = (), fontsize: float = 6.5,
               min_label_size: float = 0.0) -> None:
    poly = as_polygon(scene.room)
    ext = np.asarray(poly.exterior.coords)
    ax.add_patch(MplPolygon(ext[:-1], closed=True, facecolor="#F7F7F5",
                            edgecolor="#2B2B2B", linewidth=2.0, zorder=0))

    if show_openings:
        for op in scene.room.openings:
            c = "#2A9D8F" if op.kind == "door" else "#5B8DEF"
            ax.plot([op.p0[0], op.p1[0]], [op.p0[1], op.p1[1]],
                    color=c, linewidth=4.5, solid_capstyle="butt", zorder=3)

    # anything mounted or hanging above standing furniture -- a pendant lamp, a
    # ceiling light, a rug underfoot -- overlaps a footprint in plan while being
    # metres apart in z.  Drawing it as a solid box reads as a collision it is
    # not, so overhead and floor-mat objects are drawn as a light hatched
    # outline instead of a filled box.
    def _overhead(o):
        return o.category in ("pendant_lamp", "ceiling_lamp", "rug") or o.z >= 1.4

    for o in scene.objects:
        if not o.keep:
            continue
        corners = o.corners()
        col = _color(o.category)
        lw = 2.0 if o.oid in highlight else 0.9
        edge = "#111111" if o.oid in highlight else "#3A3A3A"
        if _overhead(o):
            ax.add_patch(MplPolygon(corners, closed=True, facecolor="none",
                                    edgecolor=col, linewidth=1.1, alpha=0.9,
                                    linestyle=(0, (3, 2)), zorder=1.5))
        else:
            ax.add_patch(MplPolygon(corners, closed=True, facecolor=col,
                                    edgecolor=edge, linewidth=lw, alpha=alpha,
                                    zorder=2 + (1 if o.category in CORE_CATEGORIES else 0)))
        if show_front and o.category not in ("rug", "pendant_lamp", "ceiling_lamp"):
            c = o.xy
            f = o.forward * (o.half[1] * 0.75 + 0.12)
            ax.arrow(c[0], c[1], f[0], f[1], head_width=0.075,
                     head_length=0.09, fc="#111111", ec="#111111",
                     linewidth=0.6, zorder=5, length_includes_head=True)
        if labels:
            # A label wider than the object it names reads as a mistake --
            # "wardrobe" spilling out of a 0.5 m box and colliding with the
            # facing arrow is what a small panel on a web page produces.  So a
            # label is only drawn when the box can hold it, and the shorthand
            # is used when the full word cannot fit.
            word = o.category.replace("_", " ")
            nlines = 1
            longest = len(word)
            need_w = longest * fontsize * 0.019        # metres at this size
            need_h = nlines * fontsize * 0.032
            avail_w = float(corners[:, 0].max() - corners[:, 0].min())
            avail_h = float(corners[:, 1].max() - corners[:, 1].min())
            fits = avail_w >= need_w and avail_h >= need_h
            short = word.split()[0][:4]
            need_s = len(short) * fontsize * 0.019
            if min_label_size <= 0 or fits:
                ax.text(o.xy[0], o.xy[1], word.replace(" ", "\n"),
                        ha="center", va="center", fontsize=fontsize,
                        linespacing=0.9, zorder=6, color="#111111")
            elif avail_w >= need_s and avail_h >= need_h:
                ax.text(o.xy[0], o.xy[1], short, ha="center", va="center",
                        fontsize=fontsize * 0.9, zorder=6, color="#111111")

    b = poly.bounds
    pad = 0.35
    ax.set_xlim(b[0] - pad, b[2] + pad)
    ax.set_ylim(b[1] - pad, b[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, pad=6)


def figure_comparison(panels: Sequence[tuple[str, Scene]], path: str | None = None,
                      per_panel: float = 3.0, labels: bool = True,
                      suptitle: str | None = None, ncols: int | None = None,
                      captions: Sequence[str] | None = None):
    """A row (or grid) of scenes drawn on a common style."""
    n = len(panels)
    ncols = ncols or n
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(per_panel * ncols, per_panel * nrows * 1.06))
    axes = np.atleast_1d(axes).ravel()
    for k, (title, scene) in enumerate(panels):
        draw_scene(axes[k], scene, title=title, labels=labels)
        if captions and k < len(captions) and captions[k]:
            axes[k].set_xlabel(captions[k], fontsize=7.5)
            axes[k].xaxis.set_label_position("bottom")
    for k in range(len(panels), len(axes)):
        axes[k].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig
