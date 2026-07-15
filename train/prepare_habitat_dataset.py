#!/usr/bin/env python3
"""train/prepare_habitat_dataset.py — build the verl RL parquet for OVMM episodes.

An RL row here is just a pointer to a Habitat episode. The real prompt (task,
memory, tool results, images) is rendered by the env server at rollout time and
never comes from the parquet, so `prompt` carries only the task text: it exists
to satisfy RLHFDataset's tokenisation/length filter and to make the dataset
readable. HabitatAgentLoop reads extra_info.episode_id and ignores raw_prompt.

Run in the verl env (needs pandas/pyarrow), from the EmbodiedAgent root:
    python train/prepare_habitat_dataset.py --split minival --out data/rl/minival.parquet
    python train/prepare_habitat_dataset.py --split train --out data/rl/train.parquet
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_OVMM = _ROOT / "data" / "datasets" / "ovmm"


def _task_text(taskmap: dict, ep_id: int) -> str:
    """Turn an OVMM episode's category triple into the natural-language task sentence.

    OVMM specifies episodes structurally (object / start receptacle / goal receptacle), but
    the agent is only ever given a sentence — so this is where the benchmark's structure is
    flattened into the single string the policy sees. Underscores become spaces because the
    raw categories are snake_case ("cutting_board") and the model reads prose better.

    Returns "" for an episode with no taskmap entry; the caller skips those rather than
    training on a task it cannot phrase.
    """
    t = taskmap.get(str(ep_id))
    if not t:
        return ""
    obj = t["object_category"].replace("_", " ")
    src = t["start_recep_category"].replace("_", " ")
    dst = t["goal_recep_category"].replace("_", " ")
    return f"Find the {obj} on the {src}, pick it up, and place it on the {dst}."


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="minival", choices=["minival", "train", "val"])
    p.add_argument("--out", default=None)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--agent_name", default="habitat")
    args = p.parse_args()

    split_file = _OVMM / f"{args.split}.json.gz"
    if not split_file.exists():
        raise SystemExit(f"{split_file} missing — run tools/convert_ovmm_episodes.py {args.split}")

    with gzip.open(split_file, "rt") as f:
        episodes = json.load(f)["episodes"]
    taskmap_file = _OVMM / f"{args.split}_taskmap.json"
    taskmap = json.loads(taskmap_file.read_text()) if taskmap_file.exists() else {}

    rows = []
    skipped = 0
    for ep in episodes:
        ep_id = int(ep["episode_id"])
        task = _task_text(taskmap, ep_id)
        if not task:
            skipped += 1
            continue
        rows.append({
            "data_source": f"habitat_ovmm_{args.split}",
            # Selects HabitatAgentLoop (registered under @register("habitat")) — this is how
            # verl knows to drive an env server rather than do a single-turn completion.
            "agent_name": args.agent_name,
            # Vestigial by design. verl's RLHFDataset insists on tokenising a prompt and
            # applying a length filter, so we give it the task sentence. The REAL prompt is
            # rendered per-turn by the env server; the agent loop never reads this field.
            "prompt": [{"role": "user", "content": task}],
            "ability": "embodied",
            # Declares that reward comes from a rule (Habitat's PDDL predicate), not a
            # learned reward model. No RM is loaded.
            "reward_model": {"style": "rule", "ground_truth": "pddl_success"},
            # The only field that actually matters: episode_id is what HabitatAgentLoop
            # sends to /reset. Everything else on this row is scaffolding.
            "extra_info": {"index": len(rows), "episode_id": ep_id, "split": args.split},
        })
        if args.max_episodes and len(rows) >= args.max_episodes:
            break

    if not rows:
        raise SystemExit(f"no episodes with a taskmap entry in {args.split}")

    out = Path(args.out) if args.out else _ROOT / "data" / "rl" / f"{args.split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} episodes -> {out}"
          + (f"  ({skipped} skipped: no taskmap entry)" if skipped else ""))


if __name__ == "__main__":
    main()
