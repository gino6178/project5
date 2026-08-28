"""Photorealistic-ish rendering of a 3D-FRONT room from its real assets.

Needed for two things the plan calls for and that box renders cannot supply:

* **an actual reference image** to feed a single-image source parser (section 6,
  experiment 14.3) -- MIDI takes RGB, not a floor plan;
* the **appearance similarity** of section 15.2, which is meaningless on
  untextured proxies.

Geometry comes from the 3D-FRONT room meshes (floor, walls, ceiling) and the
3D-FUTURE ``raw_model.obj`` of every furniture instance, placed with the same
y-up -> z-up convention the parser uses, so a rendered image and the parsed
scene describe the same room in the same frame.  Instance masks are rendered
exactly rather than segmented, which deliberately hands the parser a *favourable*
setting: any error it then makes is 3D reasoning error, not segmentation error.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import trimesh

from ..core.scene import Room, Scene
from ..data.threed_front import _yaw_from_quat, front_to_reroom
from ..geom.polygon import as_polygon, min_rotated_rect_params

__all__ = ["RoomAssets", "load_room_assets", "camera_poses", "render_room",
           "RenderResult", "available_jids", "scene_asset_coverage",
           "best_camera", "build_shell", "repose_assets",
           "render_scene_textured"]

WALL_COLOR = np.array([228, 226, 220], np.uint8)
FLOOR_COLOR = np.array([196, 178, 155], np.uint8)
CEIL_COLOR = np.array([242, 242, 240], np.uint8)


@dataclass
class RoomAssets:
    """Everything needed to render one room."""

    scene_id: str
    room: Room
    shell: list[trimesh.Trimesh] = field(default_factory=list)
    objects: list[tuple[str, trimesh.Trimesh]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.objects)


def available_jids(future_roots: list[str]) -> set[str]:
    """Model ids whose mesh is actually on disk (the release ships in parts)."""
    out: set[str] = set()
    for r in future_roots:
        if not os.path.isdir(r):
            continue
        for d in os.listdir(r):
            if os.path.exists(os.path.join(r, d, "raw_model.obj")):
                out.add(d)
    return out


def scene_asset_coverage(scene: Scene, have: set[str]) -> float:
    """Fraction of a parsed scene's objects whose mesh can be rendered."""
    jids = [o.jid for o in scene.objects if o.jid]
    if not jids:
        return 0.0
    return sum(1 for j in jids if j in have) / len(jids)


def _mesh_from_entry(m: dict, color) -> trimesh.Trimesh | None:
    v = np.asarray(m["xyz"], dtype=float).reshape(-1, 3)
    f = np.asarray(m["faces"], dtype=int).reshape(-1, 3)
    if len(v) < 3 or len(f) < 1:
        return None
    v = front_to_reroom(v)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, face_colors=np.tile(np.append(color, 255), (len(f), 1)))
    return mesh


def _load_future_mesh(jid: str, roots: list[str]) -> trimesh.Trimesh | None:
    for r in roots:
        p = os.path.join(r, jid, "raw_model.obj")
        if not os.path.exists(p):
            continue
        try:
            m = trimesh.load(p, process=False, force="mesh")
        except Exception:
            return None
        if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
            return None
        return _fix_material(m)
    return None


def _fix_material(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Make 3D-FUTURE's OBJ materials render at their real brightness.

    The shipped ``.mtl`` files carry a very dark ``Kd`` alongside the real
    texture; read literally, every sofa comes out near-black.  The texture is
    the actual appearance, so it is promoted to a PBR base-colour texture with a
    neutral factor, which is what the renderer should have been showing.
    """
    v = getattr(m, "visual", None)
    img = getattr(getattr(v, "material", None), "image", None)
    if img is None or getattr(v, "uv", None) is None:
        base = getattr(getattr(v, "material", None), "main_color", None)
        if base is not None:
            c = np.asarray(base, dtype=float)[:3]
            if float(c.max()) < 90:          # implausibly dark flat colour
                c = np.clip(c * (150.0 / max(float(c.max()), 1.0)), 0, 255)
                m.visual = trimesh.visual.ColorVisuals(
                    m, face_colors=np.tile(np.append(c.astype(np.uint8), 255),
                                           (len(m.faces), 1)))
        return m
    try:
        pbr = trimesh.visual.material.PBRMaterial(
            baseColorTexture=img,
            baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
            metallicFactor=0.0, roughnessFactor=0.85)
        m.visual = trimesh.visual.TextureVisuals(uv=v.uv, material=pbr)
    except Exception:
        pass
    return m


def load_room_assets(house_json: str, room_instanceid: str,
                     future_roots: list[str], skip_shell: bool = False,
                     only_oids: set[str] | None = None) -> RoomAssets | None:
    """Rebuild one room as real geometry, in ReRoom's z-up frame.

    ``only_oids`` restricts the furniture to what the parser actually kept, so
    the rendered instance masks and the reference scene describe the same object
    set -- otherwise a curtain the parser drops appears as an instance the
    source parser is asked to reconstruct and nothing can be matched to it.
    """
    with open(house_json) as fh:
        d = json.load(fh)
    furn = {f["uid"]: f for f in d.get("furniture", [])}
    meshes = {m["uid"]: m for m in d.get("mesh", [])}
    house = os.path.splitext(os.path.basename(house_json))[0]

    for room in d.get("scene", {}).get("room", []):
        if room.get("instanceid") != room_instanceid:
            continue
        shell, objs = [], []
        for child in room.get("children", []):
            ref = child.get("ref")
            if ref in meshes:
                if skip_shell:
                    continue
                t = meshes[ref].get("type", "")
                col = (FLOOR_COLOR if t == "Floor" else
                       CEIL_COLOR if t in ("Ceiling", "SlabTop") else
                       WALL_COLOR)
                if t in ("Floor", "WallInner", "WallOuter", "Baseboard",
                         "WallTop", "WallBottom"):
                    mm = _mesh_from_entry(meshes[ref], col)
                    if mm is not None:
                        shell.append(mm)
                continue
            f = furn.get(ref)
            if f is None or not f.get("jid"):
                continue
            if only_oids is not None and \
                    child.get("instanceid", ref) not in only_oids:
                continue
            m = _load_future_mesh(f["jid"], future_roots)
            if m is None:
                continue
            m = m.copy()
            scale = np.asarray(child.get("scale", [1, 1, 1]), dtype=float)
            m.apply_scale(scale)
            yaw = _yaw_from_quat(child.get("rot", [0, 0, 0, 1]))
            R = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
            m.apply_transform(R)
            m.apply_translation(np.asarray(child.get("pos", [0, 0, 0]), float))
            v = front_to_reroom(np.asarray(m.vertices, dtype=float))
            m = trimesh.Trimesh(vertices=v, faces=m.faces,
                                visual=m.visual, process=False)
            objs.append((child.get("instanceid", ref), m))
        if not objs:
            return None
        return RoomAssets(scene_id=f"{house}__{room_instanceid}",
                          room=Room(polygon=np.zeros((3, 2))),
                          shell=shell, objects=objs,
                          meta={"house": house, "room": room_instanceid})
    return None


@dataclass
class RenderResult:
    rgb: np.ndarray                     # (H, W, 3) uint8
    instance: np.ndarray                # (H, W) int32, -1 = background
    ids: list[str]                      # instance index -> oid
    camera: np.ndarray                  # (4, 4) camera-to-world
    coverage: dict                      # oid -> visible pixel fraction


def _look_at(eye: np.ndarray, target: np.ndarray,
             up=np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    f = target - eye
    f = f / max(np.linalg.norm(f), 1e-9)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    # pyrender/OpenGL camera looks down -z, with +x right and +y up
    M[:3, 0] = r
    M[:3, 1] = u
    M[:3, 2] = -f
    M[:3, 3] = eye
    return M


def _free_eye(room: Room, p: np.ndarray, obstacles, min_clear: float = 0.45):
    """Is this a place a photographer could actually stand?"""
    from shapely.geometry import Point
    pt = Point(float(p[0]), float(p[1]))
    if not as_polygon(room).contains(pt):
        return False
    return all(pt.distance(o) >= min_clear for o in obstacles)


def _footprints(assets: "RoomAssets"):
    """Ground-plane outlines of the furniture, for camera placement."""
    from shapely.geometry import MultiPoint
    out = []
    for _, m in assets.objects:
        v = np.asarray(m.vertices, dtype=float)[:, :2]
        if len(v) < 3:
            continue
        try:
            out.append(MultiPoint(v[:: max(len(v) // 400, 1)]).convex_hull)
        except Exception:
            continue
    return out


def camera_poses(room: Room, n: int = 4, eye_height: float = 1.60,
                 inset: float = 0.55) -> list[np.ndarray]:
    """Candidate viewpoints: stand near each corner and look across the room.

    A real-estate photograph is taken from a corner with a wide lens, which is
    also the view that sees the most furniture -- and seeing the furniture is
    the whole point when the image is going to a single-image scene parser.
    """
    poly = as_polygon(room)
    c = np.asarray(poly.centroid.coords[0])
    pts = np.asarray(poly.exterior.coords)[:-1]
    order = np.argsort(-np.linalg.norm(pts - c, axis=1))
    out = []
    for k in order[:max(n * 3, n)]:
        p = pts[k]
        d = c - p
        n_ = np.linalg.norm(d)
        if n_ < 1e-6:
            continue
        eye = p + d / n_ * min(inset, 0.45 * n_)
        target = c + (p - c) / n_ * (-0.15 * n_)
        out.append(_look_at(np.array([eye[0], eye[1], eye_height]),
                            np.array([target[0], target[1], 0.95])))
        if len(out) >= n:
            break
    return out


def best_camera(assets: "RoomAssets", room: Room, n: int = 4, width: int = 512,
                height: int = 512, min_visible_frac: float = 0.004,
                candidates: int = 12, rng=None):
    """Pick a viewpoint a photographer could stand at that sees the most objects.

    Two failure modes are ruled out explicitly, because both were observed and
    both silently corrupt everything downstream: a camera placed *inside* a
    piece of furniture (one chair filled 82 % of the frame and the dining table
    got no pixels at all), and a view that technically sees many objects but is
    dominated by one of them.  Candidates are therefore filtered for standing
    room first, then scored by how many objects are *usefully* visible.
    """
    rng = rng or np.random.default_rng(0)
    obstacles = _footprints(assets)
    poses = list(camera_poses(room, n=max(n, 6)))
    poly = as_polygon(room)
    free = poly.buffer(-0.6)
    if not free.is_empty:
        minx, miny, maxx, maxy = free.bounds
        c = np.asarray(poly.centroid.coords[0])
        for _ in range(60):
            if len(poses) >= candidates * 2:
                break
            q = rng.uniform([minx, miny], [maxx, maxy])
            if not _free_eye(room, q, obstacles):
                continue
            poses.append(_look_at(np.array([q[0], q[1], 1.60]),
                                  np.array([c[0], c[1], 0.95])))
    usable = [c for c in poses if _free_eye(room, c[:3, 3], obstacles)]
    if not usable:
        usable = poses[:1]
    best, best_score, best_res = None, -1e9, None
    for cam in usable[:candidates]:
        res = render_room(assets, cam, width, height)
        cov = np.array(list(res.coverage.values()))
        vis = cov[cov > min_visible_frac]
        if len(vis) == 0:
            continue
        # count what is usefully visible, and refuse to be impressed by a
        # single object filling the frame
        score = float(len(vis)) + float(np.clip(vis.sum(), 0, 1.5))
        if cov.max() > 0.55:
            score -= 4.0 * (cov.max() - 0.55) / 0.45
        if score > best_score:
            best, best_score, best_res = cam, score, res
    if best is None:
        best = usable[0]
        best_res = render_room(assets, best, width, height)
    return best, best_res


def render_room(assets: RoomAssets, camera: np.ndarray, width: int = 768,
                height: int = 768, yfov: float = math.radians(60.0),
                with_shell: bool = True):
    """Render RGB plus an exact instance map with pyrender (EGL, headless)."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    sc = pyrender.Scene(bg_color=[0.94, 0.94, 0.93, 1.0],
                        ambient_light=[0.80, 0.80, 0.82])
    if with_shell:
        for m in assets.shell:
            sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    nodes = []
    for oid, m in assets.objects:
        try:
            pm = pyrender.Mesh.from_trimesh(m, smooth=False)
        except Exception:
            continue
        nodes.append((oid, sc.add(pm)))

    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / height)
    cam_node = sc.add(cam, pose=camera)
    eye = camera[:3, 3]
    # a soft key from the camera plus a fill from above: enough to read shape
    # without blowing out the textures the parser has to work from
    sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.2),
           pose=camera)
    for dz, inten in ((1.4, 14.0), (-0.5, 6.0)):
        sc.add(pyrender.PointLight(color=np.ones(3), intensity=inten),
               pose=np.array([[1, 0, 0, eye[0]], [0, 1, 0, eye[1]],
                              [0, 0, 1, eye[2] + dz], [0, 0, 0, 1]], float))

    r = pyrender.OffscreenRenderer(width, height)
    rgb, depth = r.render(sc)

    # instance map: re-render each object alone against an empty scene
    inst = np.full((height, width), -1, dtype=np.int32)
    for idx, (oid, node) in enumerate(nodes):
        for _, other in nodes:
            other.mesh.is_visible = (other is node)
        d1 = r.render(sc, flags=pyrender.RenderFlags.DEPTH_ONLY)
        # tolerance has to scale with distance: a float32 depth buffer cannot
        # resolve 1 mm at 4 m, and an absolute threshold silently drops exactly
        # the large far surfaces (a dining table top) that matter most
        tol = np.maximum(1e-3, 3e-3 * np.maximum(depth, 0.0))
        inst[(d1 > 0) & (np.abs(d1 - depth) <= tol)] = idx
    for _, node in nodes:
        node.mesh.is_visible = True
    # coverage is read back off the *composited* map, not the per-object pass:
    # two coplanar surfaces both match the depth buffer, and counting them
    # separately reports instances as visible that own no pixel in the label
    # image the parser will actually be given
    coverage = {oid: float((inst == idx).mean())
                for idx, (oid, _) in enumerate(nodes)}
    r.delete()
    return RenderResult(rgb=rgb, instance=inst, ids=[o for o, _ in nodes],
                        camera=camera, coverage=coverage)


# --------------------------------------------------------------------------
# rendering a *retargeted* scene with the real assets
# --------------------------------------------------------------------------
def build_shell(room: Room, wall_height: float | None = None,
                thickness: float = 0.0) -> list[trimesh.Trimesh]:
    """Floor and walls for an arbitrary simple polygon.

    A retargeted room does not exist in 3D-FRONT, so its shell has to be made:
    the floor is the triangulated polygon, the walls are its boundary extruded
    upward.  Without this the retargeted results could only ever be drawn as
    top-down boxes, which is precisely the view that cannot answer "does this
    still look like the reference room?".
    """
    h = wall_height if wall_height is not None else room.height
    poly = as_polygon(room)
    out: list[trimesh.Trimesh] = []
    v2 = faces = None
    for engine in ("earcut", "triangle", None):
        try:
            v2, faces = (trimesh.creation.triangulate_polygon(poly, engine=engine)
                         if engine else
                         trimesh.creation.triangulate_polygon(poly))
            break
        except Exception:
            continue
    if v2 is None:
        return out
    floor = trimesh.Trimesh(
        vertices=np.column_stack([v2, np.zeros(len(v2))]), faces=faces,
        process=False)
    floor.visual = trimesh.visual.ColorVisuals(
        floor, face_colors=np.tile(np.append(FLOOR_COLOR, 255), (len(faces), 1)))
    out.append(floor)

    # Walls are wound so their normals point *into* the room.  Rendered with
    # back-face culling that gives a dollhouse view for free: the near walls
    # face away from the camera and vanish, the far walls stay and hold the
    # room together visually.
    ring = np.asarray(poly.exterior.coords)[:-1]
    verts, tris = [], []
    for k in range(len(ring)):
        a, b = ring[k], ring[(k + 1) % len(ring)]
        t = b - a
        L = float(np.linalg.norm(t))
        if L < 1e-9:
            continue
        t = t / L
        inward = np.array([-t[1], t[0]])          # CCW polygon
        i0 = len(verts)
        verts += [[a[0], a[1], 0.0], [b[0], b[1], 0.0],
                  [b[0], b[1], h], [a[0], a[1], h]]
        quad = [[i0, i0 + 1, i0 + 2], [i0, i0 + 2, i0 + 3]]
        v0, v1, v2 = (np.asarray(verts[quad[0][0]]),
                      np.asarray(verts[quad[0][1]]),
                      np.asarray(verts[quad[0][2]]))
        n = np.cross(v1 - v0, v2 - v0)
        if float(np.dot(n[:2], inward)) < 0:
            quad = [[q[0], q[2], q[1]] for q in quad]
        tris += quad
    walls = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(tris),
                            process=False)
    walls.visual = trimesh.visual.ColorVisuals(
        walls, face_colors=np.tile(np.append(WALL_COLOR, 255), (len(tris), 1)))
    out.append(walls)
    return out


def repose_assets(assets: RoomAssets, source: Scene, target: Scene,
                  bank_meshes: dict | None = None) -> RoomAssets:
    """Move each asset mesh from its reference pose to its retargeted pose.

    Objects that were *substituted* changed asset, so their mesh is looked up
    in ``bank_meshes`` when available and otherwise the reference mesh is
    rescaled -- the fallback is flagged in the returned metadata rather than
    passed off as the retrieved asset.
    """
    src_by_oid = {o.oid: o for o in source.objects}
    mesh_by_oid = {oid: m for oid, m in assets.objects}
    out, substituted_unrendered = [], []
    for t in target.objects:
        if not t.keep:
            continue
        s = src_by_oid.get(t.oid)
        m = mesh_by_oid.get(t.oid)
        if m is None:
            if t.meta.get("added"):
                continue                      # populated object, no asset yet
            continue
        m = m.copy()
        if s is None:
            out.append((t.oid, m))
            continue
        if bank_meshes and t.meta.get("substituted_from") and t.jid in bank_meshes:
            nm = bank_meshes[t.jid]
            if nm is not None:
                m = nm.copy()
        elif t.meta.get("substituted_from"):
            substituted_unrendered.append(t.oid)

        v = np.asarray(m.vertices, dtype=float)
        v[:, :2] -= s.xy                       # to the object's own frame
        v[:, 2] -= s.z
        sc = np.where(s.size > 1e-6, t.size / np.maximum(s.size, 1e-6), 1.0)
        v *= sc
        d = t.yaw - s.yaw
        c, sn = math.cos(d), math.sin(d)
        xy = v[:, :2].copy()
        v[:, 0] = xy[:, 0] * c - xy[:, 1] * sn
        v[:, 1] = xy[:, 0] * sn + xy[:, 1] * c
        v[:, :2] += t.xy
        v[:, 2] += t.z
        out.append((t.oid, trimesh.Trimesh(vertices=v, faces=m.faces,
                                           visual=m.visual, process=False)))
    return RoomAssets(scene_id=target.scene_id, room=target.room,
                      shell=build_shell(target.room), objects=out,
                      meta={"substituted_without_mesh": substituted_unrendered})


def render_scene_textured(assets: RoomAssets, room: Room, width: int = 900,
                          height: int = 640, n_cameras: int = 6,
                          elevated: bool = True):
    """Pick a readable viewpoint and render the room with its real materials."""
    if elevated:
        yfov = math.radians(52.0)
        cam = _overview_camera(room, yfov=yfov, aspect=width / height)
        return cam, render_room(assets, cam, width, height, yfov=yfov)
    return best_camera(assets, room, n=n_cameras, width=width, height=height)


def _overview_camera(room: Room, yfov: float = math.radians(52.0),
                     aspect: float = 1.4, elev_deg: float = 40.0,
                     fill: float = 1.30) -> np.ndarray:
    """A raised three-quarter view framed so the whole room fills the picture.

    Corner eye-level shots are what a parser should be fed; for *judging* a
    layout the useful view shows the whole arrangement at a readable size, so
    the distance is solved from the field of view rather than guessed.
    """
    poly = as_polygon(room)
    c = np.asarray(poly.centroid.coords[0])
    b = poly.bounds
    w, d_ = b[2] - b[0], b[3] - b[1]
    radius = 0.5 * math.hypot(w, d_)
    xfov = 2 * math.atan(math.tan(yfov / 2) * aspect)
    half = min(yfov, xfov) / 2
    dist = radius / max(math.tan(half) * fill, 1e-3)
    long_, short_, ang = min_rotated_rect_params(poly)
    view = np.array([math.cos(ang + math.pi / 2), math.sin(ang + math.pi / 2)])
    el = math.radians(elev_deg)
    eye_xy = c - view * (dist * math.cos(el))
    return _look_at(np.array([eye_xy[0], eye_xy[1], dist * math.sin(el) + 0.5]),
                    np.array([c[0], c[1], 0.75]))
