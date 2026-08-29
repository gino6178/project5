#!/usr/bin/env python
"""Why do L and T lose on out-of-floor? Measure, do not theorise.

Two hypotheses were on the table. The first -- that the halfplane containment
predicate breaks at reflex corners -- was refuted by check_concave.py: it matched
shapely exactly everywhere. The second is a loss/metric mismatch. The bnd loss is
the MEAN METRES an object pokes out; PhyScene's R_out is the FRACTION OF OBJECTS
with any pixel outside at all. A layout where many objects sit 2 cm out scores
bnd ~ 0.02, which looks solved, and R_out ~ 1.0, which is a total loss.

This reports, per shape, the rate alongside the depth distribution of the objects
that are out. If the offenders are mostly shallow, the mismatch is the cause and
the fix is to charge a rate, not a depth.
"""
import os, sys
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import importlib.util
import numpy as np
from shapely.geometry import Point

# eval_grt.py runs its whole pipeline at import; drive it in shapes mode and then
# reuse its loaded model, split and helpers rather than duplicating them.
sys.argv = [sys.argv[0], "--mode", "shapes"] + sys.argv[1:]
spec = importlib.util.spec_from_file_location("_eg", os.path.join(_ROOT, "scripts", "eval_grt.py"))
EG = importlib.util.module_from_spec(spec); spec.loader.exec_module(EG)

from reroom.geom.polygon import as_polygon, object_polygon
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph

named = EG._named_shapes()
depths = {k: [] for k in named}
rates = {k: [] for k in named}
for sd in EG.seeds:
    try:
        src = EG.test[sd]; g = build_motifs(build_scene_graph(src))
        for name, s in named.items():
            room = EG.mkroom(src.room, s)
            out = EG.run(EG.model, g, room, EG.el, EG.dev)
            rp = as_polygon(out.room)
            d = []
            for o in [o for o in out.objects if o.keep]:
                fp = object_polygon(o)
                ext = fp.difference(rp)
                if ext.is_empty:
                    d.append(0.0)
                else:
                    # deepest excursion of the footprint past the wall
                    parts = [ext] if ext.geom_type == "Polygon" else list(ext.geoms)
                    d.append(max(float(rp.exterior.distance(Point(c)))
                                 for gp in parts for c in gp.exterior.coords))
            d = np.array(d)
            if len(d):
                rates[name].append(float((d > 1e-6).mean()))
                depths[name].extend(d[d > 1e-6].tolist())
    except Exception as e:
        print("skip", sd, repr(e)[:70], flush=True)

print(f"\n  {'shape':<15}{'R_out':>8}{'n_out':>7}{'median':>9}{'p75':>8}{'p90':>8}{'max':>8}")
for name in named:
    r = float(np.mean(rates[name])) if rates[name] else float("nan")
    d = np.array(depths[name])
    if len(d) == 0:
        print(f"  {name:<15}{r:>8.3f}{0:>7}{'--':>9}{'--':>8}{'--':>8}{'--':>8}")
        continue
    print(f"  {name:<15}{r:>8.3f}{len(d):>7}{np.median(d):>9.3f}"
          f"{np.percentile(d,75):>8.3f}{np.percentile(d,90):>8.3f}{d.max():>8.3f}")
print("\ndepths are metres past the wall, over objects that are out at all")
print("DONE_ROUT")
