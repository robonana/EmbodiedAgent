#!/usr/bin/env python3
"""Convert ai-habitat/OVMM_episodes splits into RearrangeDatasetV0-compatible
episodes.json.gz that the OWMM habitat-lab fork's rearrange loader can parse.

OVMM episodes carry extra open-vocab keys (object_category, candidate_* …) that
crash `RearrangeEpisode(**episode)` (a kw_only attrs class). We strip those keys
and keep only the rearrange schema, and emit a side-car task map (episode_id ->
object/recep categories) for natural-language task derivation.

Usage:
    python tools/convert_ovmm_episodes.py minival
    python tools/convert_ovmm_episodes.py train --max 200
"""
import gzip, json, sys, glob
from pathlib import Path
import numpy as np

# keys present in OVMM episodes but NOT accepted by RearrangeEpisode.__init__
DROP = {"object_category", "start_recep_category", "goal_recep_category",
        "candidate_objects", "candidate_objects_hard",
        "candidate_start_receps", "candidate_goal_receps"}

HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "data" / "datasets" / "ovmm"
SNAP = next(Path.home().glob(
    ".cache/huggingface/hub/datasets--ai-habitat--OVMM_episodes/snapshots/*"))


def load_split(split: str):
    """Return (wrapper_dict, episodes_list, transforms). For train, merge content/*.json.gz."""
    top = SNAP / split / "episodes.json.gz"
    d = json.load(gzip.open(top))
    eps = d.get("episodes", [])
    if split == "train" and not eps:
        # train ships per-scene content files
        for f in sorted(glob.glob(str(SNAP / "train" / "content" / "*.json.gz"))):
            eps.extend(json.load(gzip.open(f)).get("episodes", []))
    transforms = np.load(SNAP / split / "transformations.npy")  # (N, 3, 4)
    return d, eps, transforms


def _to_4x4(mat3x4):
    """Append the homogeneous row so the rearrange sim can build an mn.Matrix4."""
    m = np.asarray(mat3x4, dtype=float)
    out = np.eye(4)
    out[:3, :4] = m
    return out.tolist()


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "minival"
    cap = None
    if "--max" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--max") + 1])

    wrapper, eps, T = load_split(split)
    if cap:
        eps = eps[:cap]
    print(f"[{split}] {len(eps)} episodes; transforms {T.shape}")

    task_map = {}
    clean_eps = []
    for ep in eps:
        eid = str(ep["episode_id"])
        task_map[eid] = {
            "object_category": ep.get("object_category", ""),
            "start_recep_category": ep.get("start_recep_category", ""),
            "goal_recep_category": ep.get("goal_recep_category", ""),
            "scene_id": ep.get("scene_id", ""),
        }
        ce = {k: v for k, v in ep.items() if k not in DROP}
        # OVMM stores rigid_objs transforms as int indices into transformations.npy;
        # dereference to inline 4x4 matrices for the rearrange json loader.
        ce["rigid_objs"] = [[name, _to_4x4(T[idx])] for name, idx in ce["rigid_objs"]]
        # The PDDL problem (pddl_single_agent_man) names the target `any_targets|N`,
        # but OVMM labels it `hab2|N`. Remap the goal-label prefix so PDDL binding
        # finds the entity (sim reads ep.info["object_labels"]).
        info = dict(ce.get("info", {}) or {})
        if "object_labels" in info:
            info["object_labels"] = {
                k: f"any_targets|{v.split('|')[-1]}"
                for k, v in info["object_labels"].items()
            }
        ce["info"] = info
        clean_eps.append(ce)

    out = dict(wrapper)
    out["episodes"] = clean_eps

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split}.json.gz"
    with gzip.open(out_path, "wt") as f:
        json.dump(out, f)
    map_path = OUT_DIR / f"{split}_taskmap.json"
    map_path.write_text(json.dumps(task_map))
    print(f"wrote {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")
    print(f"wrote {map_path}  ({len(task_map)} tasks)")


if __name__ == "__main__":
    main()
