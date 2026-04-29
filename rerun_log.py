"""
rerun_log.py — Rerun viewer integration.

RERUN_ENABLED is a module-level bool set to True by navigate.main() when
--rerun is passed.  All other modules import this flag from here rather than
carrying their own copy, so a single assignment in main() propagates everywhere.
"""

import numpy as np
import torch

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    rr = None
    RERUN_AVAILABLE = False
    print("Rerun not available — install with: pip install rerun-sdk")

# Set to True in navigate.main() when --rerun is passed.
RERUN_ENABLED: bool = False


def log_cameras_rerun(obs: dict) -> None:
    """Log sensor camera images to Rerun from the step observation dict.

    Only robot-mounted sensors (fetch_head, arm cameras) are logged, not
    ManiSkill's fixed world-space 'base_camera'.
    """
    if not RERUN_ENABLED:
        return
    try:
        sensor_data = obs.get("sensor_data", {})
        for cam_name, cam_data in sensor_data.items():
            img_data = (cam_data.get("rgb") if cam_data.get("rgb") is not None
                        else cam_data.get("Color"))
            if img_data is None:
                continue
            if torch.is_tensor(img_data):
                img_data = img_data.cpu().numpy()
            img = np.array(img_data).squeeze()
            if img.ndim == 4:
                img = img[0]
            img = img[..., :3] if img.shape[-1] == 4 else img
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            if "arm_camera" in cam_name:
                img = np.rot90(img, k=3)
            rr.log(f"cameras/{cam_name}", rr.Image(img))
    except Exception as _e:
        print(f"[rerun] camera log error: {_e}")
