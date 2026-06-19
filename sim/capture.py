from __future__ import annotations

"""
capture.py — Navigation frame capture.

_capture_nav_frame   Extract the fetch_head RGB frame from a step observation.
NavCaptureWorker     Background thread that writes color PNGs + robot poses to disk.
"""

import math
import queue as _queue_mod
import threading

import numpy as np
import torch


def _capture_nav_frame(obs: dict) -> np.ndarray | None:
    """Return fetch_head rgb uint8 HxWx3 from the step observation dict, or None."""
    try:
        cam_data = obs["sensor_data"]["fetch_head"]
        rgb = (cam_data.get("rgb") if cam_data.get("rgb") is not None
               else cam_data.get("Color"))
        if rgb is None:
            return None
        if torch.is_tensor(rgb):
            rgb = rgb.cpu().numpy()
        rgb = np.array(rgb).squeeze()
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb = rgb[..., :3] if rgb.shape[-1] == 4 else rgb
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        return rgb
    except Exception:
        return None


def save_current_frame(
    rgb: np.ndarray,
    output_dir: str,
    step_idx: int,
    tag: str = "frame",
) -> str:
    """Save an RGB numpy array as a PNG and return the file path."""
    from PIL import Image as _PILImage
    import pathlib
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = str(out / f"{step_idx:06d}_{tag}.png")
    _PILImage.fromarray(rgb).save(path)
    return path


def crop_image(
    image_path: str,
    bbox: list,
    output_dir: str,
    step_idx: int = 0,
) -> str:
    """Crop image_path to bbox [x1, y1, x2, y2] and save. Returns crop path."""
    from PIL import Image as _PILImage
    import pathlib
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    img  = _PILImage.open(image_path).convert("RGB")
    crop = img.crop((x1, y1, x2, y2))
    path = str(out / f"{step_idx:06d}_crop_{x1}_{y1}_{x2}_{y2}.png")
    crop.save(path)
    return path


class NavCaptureWorker:
    """
    Background thread that saves nav-capture frames to the scene dataset.

    Layout
    ------
    <out_dir>/color/<idx:06d>.png
    <out_dir>/robot_xy/<idx:06d>.txt   — [x  y  yaw_rad]

    Frame indices continue from the highest existing color/ frame so scans
    accumulate seamlessly across multiple navigation runs.
    """

    def __init__(self, out_dir: str):
        import pathlib
        self._out = pathlib.Path(out_dir)
        for d in ("color", "robot_xy"):
            (self._out / d).mkdir(parents=True, exist_ok=True)

        existing = sorted((self._out / "color").glob("*.png"))
        self._next_idx = int(existing[-1].stem) + 1 if existing else 0
        print(f"  [NavCapture] Dataset: {self._out}  "
              f"next frame idx={self._next_idx}")

        self._q      = _queue_mod.Queue()
        self.saved   = 0
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="NavCapture")
        self._thread.start()

    def enqueue(self, rgb: np.ndarray,
                robot_xy: np.ndarray, robot_yaw: float) -> int:
        """Queue a frame for async save; returns the assigned frame index."""
        idx = self._next_idx
        self._next_idx += 1
        self._q.put((idx, rgb.copy(), robot_xy.copy(), float(robot_yaw)))
        return idx

    def flush(self) -> None:
        """Block until all queued frames are written to disk."""
        self._q.join()

    def stop(self) -> int:
        """Flush and stop the worker thread; returns total frames saved."""
        self._q.put(None)
        self._thread.join(timeout=60.0)
        return self.saved

    def _worker(self):
        from PIL import Image as _PILImage
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                break
            idx, rgb, robot_xy, robot_yaw = item
            try:
                stem = f"{idx:06d}"
                _PILImage.fromarray(rgb).save(
                    str(self._out / "color" / f"{stem}.png"))
                np.savetxt(
                    str(self._out / "robot_xy" / f"{stem}.txt"),
                    np.array([[robot_xy[0], robot_xy[1], robot_yaw]]))
                self.saved += 1
                print(f"  [NavCapture] frame {stem}  "
                      f"xy=({robot_xy[0]:.2f},{robot_xy[1]:.2f})  "
                      f"yaw={math.degrees(robot_yaw):.0f}°", flush=True)
            except Exception as _e:
                print(f"  [NavCapture] ERROR saving frame {idx}: {_e}")
            finally:
                self._q.task_done()
