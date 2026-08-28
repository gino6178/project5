"""Lightweight 3D scene rendering.

Used for the paper's perspective figures and as the image source for the
*auxiliary* appearance metric of section 15.2.  Objects are drawn as oriented
boxes shaded by category; when real 3D-FUTURE meshes are available their
product image is used for the object-matched appearance score instead, which is
both cheaper and more faithful than rendering untextured proxies.

Deliberately dependency-light: matplotlib's 3D backend needs no GL context, no
display and no external renderer, so figures reproduce anywhere.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..core.scene import ObjectInstance, Scene
from ..geom.polygon import as_polygon
from .topdown import _color

__all__ = ["render_scene3d", "CANONICAL_VIEWS"]

CANONICAL_VIEWS = ((28.0, -60.0), (28.0, 30.0), (28.0, 120.0), (28.0, 210.0))


def _box_faces(o: ObjectInstance) -> list[np.ndarray]:
    c = o.corners()
    z0, z1 = o.z, o.top
    lo = [np.array([p[0], p[1], z0]) for p in c]
    hi = [np.array([p[0], p[1], z1]) for p in c]
    faces = [np.array(hi), np.array(lo[::-1])]
    for k in range(4):
        a, b = k, (k + 1) % 4
        faces.append(np.array([lo[a], lo[b], hi[b], hi[a]]))
    return faces


def _shade(color: str, factor: float) -> tuple:
    rgb = matplotlib.colors.to_rgb(color)
    return tuple(float(np.clip(c * factor, 0, 1)) for c in rgb)


def render_scene3d(scene: Scene, path: str | None = None,
                   view: tuple[float, float] = CANONICAL_VIEWS[0],
                   figsize: float = 4.0, show_walls: bool = True,
                   title: str | None = None, dpi: int = 150):
    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_subplot(111, projection="3d")
    poly = as_polygon(scene.room)
    ring = np.asarray(poly.exterior.coords)[:-1]
    h = scene.room.height

    ax.add_collection3d(Poly3DCollection(
        [np.column_stack([ring, np.zeros(len(ring))])],
        facecolor="#EDEDE8", edgecolor="#8A8A8A", linewidths=0.7, zorder=0))
    if show_walls:
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            quad = np.array([[a[0], a[1], 0], [b[0], b[1], 0],
                             [b[0], b[1], h], [a[0], a[1], h]])
            ax.add_collection3d(Poly3DCollection(
                [quad], facecolor="#FAFAF8", edgecolor="#C8C8C4",
                linewidths=0.5, alpha=0.32))

    for o in scene.objects:
        if not o.keep:
            continue
        base = _color(o.category)
        for fi, f in enumerate(_box_faces(o)):
            shade = 1.0 if fi == 0 else (0.72 if fi == 1 else 0.86 - 0.05 * fi)
            ax.add_collection3d(Poly3DCollection(
                [f], facecolor=_shade(base, shade), edgecolor="#333333",
                linewidths=0.35, alpha=0.97))

    b = poly.bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    r = max(b[2] - b[0], b[3] - b[1]) / 2 + 0.4
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(0, max(h, 2 * r * 0.55))
    try:
        ax.set_box_aspect((1, 1, 0.55))
    except Exception:
        pass
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.1)
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        return path
    return fig
