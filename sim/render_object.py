from __future__ import annotations

"""
Render views of a ReplicaCAD object by orbiting the camera around it.
Produces clean white-background, object-centred images for retrieval queries.

Usage:
    python render_replicacad_object.py --object_type frl_apartment_bowl_01
    python render_replicacad_object.py --object_type bowl_01   # short name OK
    python render_replicacad_object.py --n_images 6 --output_dir ./query_images
    python render_replicacad_object.py --list

Via shell script:
    ./query_robocasa_object.sh render_replicacad --object_type bowl_01
    ./query_robocasa_object.sh render_replicacad --list

Assets are read from the ManiSkill data directory:
    ~/.maniskill/data/scene_datasets/replica_cad_dataset/objects/

Why this exists: `retrieve_memory` accepts an IMAGE as its query, not just text. To ask
"where did I see this bowl?", something has to produce a clean reference picture of *that
specific* bowl. This renders the object's .glb in isolation — white background, tightly
cropped, several viewpoints — so the resulting images embed cleanly and are not polluted by
scene context the way a crop from a cluttered frame would be.

Rendered with Open3D's offscreen renderer, entirely independently of the simulator; it never
loads a scene, only a single asset mesh.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import imageio

# ── asset discovery ───────────────────────────────────────────────────────────

def _objects_dir() -> Path:
    """Return the ReplicaCAD objects directory (raises if not found)."""
    # Try mani_skill first; fall back to default install path.
    try:
        from mani_skill import ASSET_DIR
        d = Path(ASSET_DIR) / "scene_datasets/replica_cad_dataset/objects"
    except ImportError:
        d = Path.home() / ".maniskill/data/scene_datasets/replica_cad_dataset/objects"
    if not d.is_dir():
        raise FileNotFoundError(
            f"ReplicaCAD objects directory not found: {d}\n"
            "Run: python -m mani_skill.utils.download_asset ReplicaCAD_SceneManipulation-v1"
        )
    return d


def list_objects() -> list[str]:
    """Return sorted list of all ReplicaCAD object names (without .glb suffix)."""
    return sorted(p.stem for p in _objects_dir().glob("*.glb"))


def resolve_object_path(name: str) -> Path:
    """Resolve any of these name formats to its .glb path:
      - '63'                                   (index into sorted object list)
      - 'bowl_01'                              (short)
      - 'frl_apartment_bowl_01'                (full stem)
      - 'env-0_objects/frl_apartment_bowl_01-105'  (ep_meta.json format)
    """
    d = _objects_dir()

    # Numeric index → look up in sorted list.
    # Convenience for the CLI: `--list` prints indices, so you can render #63 without
    # typing the full asset name. Must match --list's sort order exactly.
    if name.lstrip("-").isdigit():
        objs = sorted(p.stem for p in d.glob("*.glb"))
        idx = int(name)
        if not (0 <= idx < len(objs)):
            raise ValueError(f"Index {idx} out of range (0–{len(objs)-1})")
        name = objs[idx]

    # Strip ep_meta path prefix and instance-ID suffix:
    #   'env-0_objects/frl_apartment_bowl_01-105' → 'frl_apartment_bowl_01'
    # This lets an object name copied straight out of an episode's ep_meta.json be passed
    # in verbatim — the scene refers to a specific *instance* of an asset, but the asset
    # file on disk is shared by all instances of it.
    stem = name
    if "/" in stem:
        stem = stem.rsplit("/", 1)[-1]   # drop 'env-0_objects/' prefix
    import re
    stem = re.sub(r"-\d+$", "", stem)    # drop '-105' instance suffix

    # Try exact stem match
    p = d / f"{stem}.glb"
    if p.exists():
        return p
    # Try with frl_apartment_ prefix — nearly every ReplicaCAD asset carries it, so users
    # (and the VLM) reasonably say "bowl_01" and expect it to resolve.
    p = d / f"frl_apartment_{stem}.glb"
    if p.exists():
        return p

    available = [x.stem for x in d.glob("*.glb")]
    raise ValueError(
        f"Object '{name}' (resolved stem: '{stem}') not found in {d}.\n"
        f"Run --list to see all {len(available)} available objects."
    )


# ── rendering ─────────────────────────────────────────────────────────────────

def _look_at_pose(eye: np.ndarray, target: np.ndarray,
                  up: np.ndarray = None):
    """Return (eye, up) ready for open3d camera.look_at.

    Re-orthogonalises the up vector: the caller's `up` is only a *hint* at which way is
    up, and it is generally not perpendicular to the viewing direction. Crossing twice
    (fwd × up → right, then right × fwd → true up) produces the actual orthogonal basis
    vector the camera needs.
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])   # Z-up, matching the orbit below
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    norm = np.linalg.norm(right)
    if norm < 1e-6:                     # degenerate: eye directly above target
        # fwd is parallel to up, so their cross product vanishes and "right" is undefined.
        # Pick any other axis as the hint and redo it.
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
        norm = np.linalg.norm(right)
    right /= norm
    actual_up = np.cross(right, fwd)
    return eye, actual_up


def _orbit_eyes(center: np.ndarray, radius: float,
                n: int, elev_deg: float = 30.0) -> list[np.ndarray]:
    """Generate n camera eye positions evenly spaced in azimuth at a fixed elevation.

    A ring around the object at constant elevation — standard spherical-to-Cartesian, with
    cos(elev) shrinking the horizontal radius as the camera rises. A modest elevation (30°
    by default) looks slightly down at the object, which is roughly the angle the robot's
    head camera sees it from, so the query images match the indexed frames reasonably well.
    """
    eyes = []
    elev = math.radians(elev_deg)
    for i in range(n):
        az = 2 * math.pi * i / n     # evenly spaced around the full circle
        x = center[0] + radius * math.cos(az) * math.cos(elev)
        y = center[1] + radius * math.sin(az) * math.cos(elev)
        z = center[2] + radius * math.sin(elev)
        eyes.append(np.array([x, y, z]))
    return eyes


def _crop_to_object(img: np.ndarray, pad_frac: float = 0.08) -> np.ndarray | None:
    """
    Detect non-white pixels, crop tightly, pad to square, return uint8 H×W×3.
    Returns None if the object is not visible.

    Framing matters more than it might seem. The embedding model sees a fixed-size square,
    so an object rendered small in the corner of the frame occupies few tokens and embeds
    poorly. Cropping to the object and letting it fill the image gives the encoder the most
    signal per pixel. Padding to a square (rather than resizing a rectangle) preserves the
    aspect ratio, so a tall bottle does not get squashed into a squat one.
    """
    # Background is set to pure white by the renderer, so anything not-quite-white is the
    # object. The ≥250 threshold (not ==255) tolerates antialiasing at the object's edge.
    white = np.all(img >= 250, axis=2)   # background mask
    obj_mask = ~white

    # <0.1% of pixels: the object is off-screen, or so small the render is useless. Reject
    # rather than return a crop of a few stray antialiased pixels.
    if obj_mask.sum() < img.shape[0] * img.shape[1] * 0.001:
        return None                       # object too small / not visible

    # Tight bounding box: first and last row/column containing any object pixel.
    rows = np.where(obj_mask.any(axis=1))[0]
    cols = np.where(obj_mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]

    # Breathing room, proportional to the object's size so it looks the same at any scale;
    # at least 4 px so a tiny object still gets a margin.
    h_box = r1 - r0 + 1
    w_box = c1 - c0 + 1
    pad_px = max(4, int(max(h_box, w_box) * pad_frac))

    H, W = img.shape[:2]
    r0 = max(0, r0 - pad_px)
    r1 = min(H - 1, r1 + pad_px)
    c0 = max(0, c0 - pad_px)
    c1 = min(W - 1, c1 + pad_px)

    crop = img[r0:r1+1, c0:c1+1]

    # Pad to square with white — centred, so the object stays in the middle of the frame
    # and the padding is indistinguishable from the rendered background.
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    square = np.full((side, side, 3), 255, dtype=np.uint8)
    top  = (side - ch) // 2
    left = (side - cw) // 2
    square[top:top+ch, left:left+cw] = crop

    return square


def render(args):
    """Orbit the camera around one object and save n cropped, square views."""
    import open3d as o3d
    from PIL import Image

    glb_path = resolve_object_path(args.object_type)
    print(f"Rendering: {glb_path.name}")

    model = o3d.io.read_triangle_model(str(glb_path))

    # Render at 2× the output resolution, then downsample after cropping. The crop discards
    # most of the frame, so rendering at 1× would leave the object with too few pixels to
    # resize *up* to the target size without visible softness. Supersampling also cleans up
    # the edges for free.
    render_w = args.width * 2
    render_h = args.height * 2

    renderer = o3d.visualization.rendering.OffscreenRenderer(render_w, render_h)
    # Pure white — _crop_to_object keys off exactly this to find the object.
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.add_model("obj", model)

    # Lighting. Bright, with strong indirect light: the aim is an evenly-lit product shot in
    # which the whole object is legible, not a dramatic one with half of it in shadow — the
    # embedding should capture the object, not the lighting.
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_sun_light([0.5, -1.0, -0.5], [1.5, 1.5, 1.5], 100000)
    renderer.scene.scene.set_indirect_light_intensity(50000)

    # Object bounds → orbit radius. Scaling the bounding-sphere diameter means a mug and a
    # sofa are both framed correctly without per-object tuning.
    bounds = renderer.scene.bounding_box
    center = np.asarray(bounds.get_center())
    extent = np.asarray(bounds.get_extent())
    radius = np.linalg.norm(extent) * args.distance_scale

    fov_deg = args.fov
    # Near plane at 1 mm: assets are small (a few cm), and a conventional 0.1 m near plane
    # would clip the object away entirely.
    renderer.scene.camera.set_projection(
        fov_deg, render_w / render_h, 0.001, 100.0,
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )

    eyes = _orbit_eyes(center, radius, args.n_images, elev_deg=args.elevation)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    stem = glb_path.stem               # e.g. frl_apartment_bowl_01
    short = stem.replace("frl_apartment_", "")  # e.g. bowl_01

    for i, eye in enumerate(eyes):
        eye_arg, up_arg = _look_at_pose(eye, center)
        renderer.scene.camera.look_at(center.tolist(), eye_arg.tolist(), up_arg.tolist())

        img_o3d = renderer.render_to_image()
        img_np = np.asarray(img_o3d)   # H×W×3, uint8

        cropped = _crop_to_object(img_np)
        if cropped is None:
            # One bad viewpoint (object edge-on, or occluded by its own geometry) doesn't
            # invalidate the others — skip it and keep orbiting.
            print(f"  View {i}: object not visible, skipping")
            continue

        # Resize to target resolution (down from the 2× supersampled render).
        out_img = np.array(
            Image.fromarray(cropped).resize((args.width, args.height), Image.LANCZOS)
        )

        fname = f"{stem}_{i}.png"   # full name e.g. frl_apartment_bowl_01_0.png
        out_path = out_dir / fname
        imageio.imwrite(str(out_path), out_img)
        saved.append(str(out_path))
        print(f"  Saved {out_path}")

    renderer.scene.remove_geometry("obj")
    print(f"\n{len(saved)}/{args.n_images} images saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Render ReplicaCAD object views for retrieval queries"
    )
    p.add_argument("--object_type", type=str, default=None,
                   help="Object name, e.g. bowl_01 or frl_apartment_bowl_01")
    p.add_argument("--list", action="store_true",
                   help="List all available object names and exit")
    p.add_argument("--n_images", type=int, default=4,
                   help="Number of orbit viewpoints to save (default: 4)")
    p.add_argument("--elevation", type=float, default=30.0,
                   help="Camera elevation angle in degrees (default: 30)")
    p.add_argument("--distance_scale", type=float, default=0.85,
                   help="Camera distance as fraction of object bounding-sphere diameter (default: 0.85)")
    p.add_argument("--fov", type=float, default=60.0,
                   help="Vertical field of view in degrees (default: 60)")
    p.add_argument("--output_dir",
                   default=str(Path.home() / "Projects/concept-graphs/query_images"),
                   help="Directory to save rendered images")
    p.add_argument("--width",  type=int, default=512)
    p.add_argument("--height", type=int, default=512)

    args = p.parse_args()

    if args.list:
        objs = list_objects()
        print(f"ReplicaCAD objects ({len(objs)} total):")
        for i, name in enumerate(objs):
            short = name.replace("frl_apartment_", "")
            print(f"  [{i:3d}]  {short:40s}  ({name})")
        return

    if args.object_type is None:
        p.error("--object_type is required (or --list)")

    render(args)


if __name__ == "__main__":
    main()


# ── async helper for use from run.py ─────────────────────────────────────────
#
# Rendering runs in a SEPARATE PROCESS, not just a separate thread, and that is the whole
# point of this section. Open3D's offscreen renderer and the simulator (SAPIEN/Vulkan) both
# want to own the GPU context, and initialising Open3D inside a live sim process can crash
# it outright. Shelling out to a fresh interpreter gives the renderer its own context, which
# dies cleanly with the subprocess.
#
# That subprocess may also need a *different conda environment* — the renderer needs Open3D,
# which the sim env does not necessarily have — hence the interpreter search below.

import os as _os
import threading as _threading

_SCRIPT_PATH = _os.path.abspath(__file__)   # re-invoke this same file as a CLI

# First interpreter that exists wins. Ordered by preference: the concept-graphs envs are
# known to carry Open3D; the maniskill env is the last-resort fallback (and is where we
# already are, so it works if Open3D happens to be installed).
_cg_py     = _os.path.expanduser("~/miniconda3/envs/cg/bin/python")
_cgraph_py = _os.path.expanduser("~/miniconda3/envs/conceptgraph/bin/python")
_ms_py     = _os.path.expanduser("~/miniconda3/envs/maniskill/bin/python")
RENDER_PYTHON = (
    _cg_py     if _os.path.exists(_cg_py)
    else _cgraph_py if _os.path.exists(_cgraph_py)
    else _ms_py
)


def render_object_async(
    object_name: str,
    output_dir: str,
    on_done: "callable[[list[str]], None]",
    on_error: "callable[[str], None]",
) -> _threading.Thread:
    """Render object views in a background thread.

    on_done(image_paths) called on success.
    on_error(message)    called on failure.
    Returns the daemon Thread (already started).

    Thread + subprocess: the thread exists only so the caller (the interactive viewer) is
    not blocked for the seconds the render takes; the subprocess is what isolates the GPU
    context. Callbacks fire ON THE WORKER THREAD, so anything they touch must be safe to
    touch from there.
    """
    def _worker():
        import subprocess as _sp
        try:
            rc = _sp.run(
                [RENDER_PYTHON, _SCRIPT_PATH,
                 "--object_type", object_name,
                 "--output_dir",  output_dir],
                # capture_output=False: let the child's logs stream to our console, so a
                # render failure is visible rather than swallowed.
                capture_output=False, text=True,
            )
            if rc.returncode == 0:
                imgs = sorted(
                    f for f in _os.listdir(output_dir)
                    if f.lower().endswith(".png")
                )
                if imgs:
                    on_done([_os.path.join(output_dir, f) for f in imgs])
                else:
                    # Exit code 0 but nothing produced — every viewpoint was rejected by
                    # _crop_to_object. Distinguish it from a hard failure.
                    on_error(f"Render OK but no images in {output_dir}")
            else:
                on_error(f"Render failed (rc={rc.returncode})")
        except Exception as e:
            # Includes the interpreter not existing at all — report, never raise into the
            # thread's top level where it would vanish silently.
            on_error(f"Render error: {e}")

    t = _threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
