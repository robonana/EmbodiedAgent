"""
rerun_log.py — Rerun viewer integration.

RERUN_ENABLED is a module-level bool set to True by navigate.main() when
--rerun is passed.  All other modules import this flag from here rather than
carrying their own copy, so a single assignment in main() propagates everywhere.

Purely a visualisation side-channel: nothing here feeds the agent, and every function is a
no-op when Rerun is disabled, so call sites never need to guard.
"""

import numpy as np
import torch

# Rerun is an optional dependency. Absent, the module still imports and every log call
# becomes a no-op — a missing viewer must never stop a run.
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    rr = None
    RERUN_AVAILABLE = False
    print("Rerun not available — install with: pip install rerun-sdk")

# Set to True in navigate.main() when --rerun is passed.
# Deliberately a module-level global rather than a value threaded through call sites: it is
# read from deep inside the control loops, and `from rerun_log import RERUN_ENABLED` would
# copy the value at import time, so importers must reference it as `rerun_log.RERUN_ENABLED`
# (or, as here, only this module reads it).
RERUN_ENABLED: bool = False


def log_cameras_rerun(obs: dict) -> None:
    """Log sensor camera images to Rerun from the step observation dict.

    Only robot-mounted sensors (fetch_head, arm cameras) are logged, not
    ManiSkill's fixed world-space 'base_camera'.

    The normalisation below mirrors sim/capture.py::_capture_nav_frame — same tensor/numpy,
    batch-dim, RGBA and float-vs-uint8 variance to flatten, for the same reason (what the
    renderer hands back varies with backend and config).
    """
    if not RERUN_ENABLED:
        return
    try:
        sensor_data = obs.get("sensor_data", {})
        for cam_name, cam_data in sensor_data.items():
            img_data = (cam_data.get("rgb") if cam_data.get("rgb") is not None
                        else cam_data.get("Color"))    # newer vs older ManiSkill key
            if img_data is None:
                continue
            if torch.is_tensor(img_data):
                img_data = img_data.cpu().numpy()
            img = np.array(img_data).squeeze()
            if img.ndim == 4:
                img = img[0]                            # (B,H,W,C) → first env
            img = img[..., :3] if img.shape[-1] == 4 else img   # drop alpha
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            # The arm camera is physically mounted rotated on the robot, so its raw frame
            # arrives sideways. k=3 (270° CCW == 90° CW) puts it upright for a human viewer.
            # Cosmetic only — the agent sees the unrotated frame.
            if "arm_camera" in cam_name:
                img = np.rot90(img, k=3)
            # Entity path groups all cameras under one collapsible node in the viewer.
            rr.log(f"cameras/{cam_name}", rr.Image(img))
    except Exception as _e:
        # Logging must never break the sim loop it is called from.
        print(f"[rerun] camera log error: {_e}")
