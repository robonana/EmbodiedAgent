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

Run once per split. Its outputs (data/datasets/ovmm/{split}.json.gz and
{split}_taskmap.json) are what run_ovmm_embodied.py and train/prepare_habitat_dataset.py
both consume, so nothing downstream works until this has run.

Three incompatibilities are fixed here; each is documented at its site below:
  1. open-vocab keys that crash the attrs-based RearrangeEpisode constructor  (DROP)
  2. rigid_objs transforms stored as indices into a side-car .npy             (_to_4x4)
  3. PDDL goal labels using OVMM's naming rather than the task's              (object_labels)
"""
import gzip, json, sys, glob
from pathlib import Path
import numpy as np

# keys present in OVMM episodes but NOT accepted by RearrangeEpisode.__init__
# RearrangeEpisode is a kw_only attrs class, so an unexpected keyword is a hard TypeError
# rather than being ignored — every one of these must go. They are not lost: the three
# *_category fields are what the side-car taskmap preserves, and they become the task
# sentence. The candidate_* lists (alternative valid objects/receptacles) are ground truth
# for OVMM's own evaluator and are genuinely unused by us.
DROP = {"object_category", "start_recep_category", "goal_recep_category",
        "candidate_objects", "candidate_objects_hard",
        "candidate_start_receps", "candidate_goal_receps"}

HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "data" / "datasets" / "ovmm"
# Resolve the HF dataset cache snapshot (its directory name is a commit hash, hence glob).
# next() raises StopIteration if the dataset was never downloaded — a loud, early failure,
# which is what we want.
SNAP = next(Path.home().glob(
    ".cache/huggingface/hub/datasets--ai-habitat--OVMM_episodes/snapshots/*"))


def load_split(split: str):
    """Return (wrapper_dict, episodes_list, transforms). For train, merge content/*.json.gz.

    `wrapper_dict` is the whole top-level object; we keep it so the output preserves any
    sibling metadata alongside "episodes".

    The train split ships its episodes sharded per scene under content/ with an empty
    top-level list, while minival/val inline them — hence the merge branch.
    """
    top = SNAP / split / "episodes.json.gz"
    d = json.load(gzip.open(top))
    eps = d.get("episodes", [])
    if split == "train" and not eps:
        # train ships per-scene content files
        for f in sorted(glob.glob(str(SNAP / "train" / "content" / "*.json.gz"))):
            eps.extend(json.load(gzip.open(f)).get("episodes", []))
    # Object poses live OUT of line, in a single big array indexed by the episodes.
    transforms = np.load(SNAP / split / "transformations.npy")  # (N, 3, 4)
    return d, eps, transforms


def _to_4x4(mat3x4):
    """Append the homogeneous row so the rearrange sim can build an mn.Matrix4.

    OVMM stores a 3×4 (rotation | translation) matrix; Magnum's Matrix4 wants a full 4×4.
    Starting from an identity means the [0,0,0,1] bottom row is already correct.
    """
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
        # Rescue the semantic fields before DROP discards them. This side-car map is the
        # ONLY place the object/receptacle categories survive, and it is what
        # prepare_habitat_dataset.py turns into the natural-language task sentence. Keyed by
        # string id, because JSON object keys are strings.
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
        #
        # Without this the PDDL goal never binds to any entity and pddl_success is
        # permanently 0 — the episode runs perfectly and always scores as a failure, which
        # in RL means a reward signal of exactly zero. Only the prefix changes; the trailing
        # index (split on "|", take the last part) identifies WHICH target and is preserved.
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
