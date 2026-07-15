#!/usr/bin/env python3
"""
compare_results.py — Compare EmbodiedAgent vs OWMM-Agent on the same benchmark.

Reads:
  1. EmbodiedAgent results from runs/owmm_embodied/results.csv  (run_owmm_embodied.py)
  2. OWMM-Agent results from OWMM-Agent eval logs               (episodic_eval_owmmvlm.py)

Outputs a side-by-side comparison table.

Usage:
    python compare_results.py
    python compare_results.py --embodied runs/owmm_embodied/results.csv \\
                              --owmm     OWMM-Agent/sim/habitat-lab/eval_in_sim_info/sat_TEST_YCB_30scene_head_rgb

The two frameworks report the same underlying Habitat metrics but in completely different
formats — ours in a CSV column-per-metric, theirs scraped out of human-readable text logs, one
directory per episode. Most of this file is reconciling those two shapes into one table.

Naming gotcha that runs through the whole module: our CSV prefixes Habitat's metrics with
`hab_` (e.g. hab_pddl_success) while OWMM's logs do not (pddl_success). The _avg() fallback
mechanism exists purely to paper over that, and the per-episode table hard-codes both spellings.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


_HERE = Path(__file__).resolve().parent


# ── OWMM-Agent log parsing ────────────────────────────────────────────────────

def _parse_owmm_log(output_log: Path) -> Optional[Dict]:
    """Parse a single OWMM-Agent episode output.log → metric dict.

    OWMM emits no machine-readable results file, so we scrape its console log for lines of the
    form "Average episode <metric>: <value>". Each episode's log lives in a directory NAMED
    for its episode id, which is the only place that id appears — hence recovering it from the
    parent directory name.

    errors="replace" because these logs contain progress bars and terminal escapes that are
    not always valid UTF-8. Returns None when nothing parsed, so a crashed episode's log is
    skipped rather than counted as a zero.
    """
    if not output_log.exists():
        return None
    text = output_log.read_text(errors="replace")
    metrics = {}
    for line in text.splitlines():
        m = re.search(r"Average episode (\S+):\s+([\d.]+)", line)
        if m:
            # Later lines win — the log prints running averages, so the final one is the
            # episode's actual result.
            metrics[m.group(1)] = float(m.group(2))
    if not metrics:
        return None
    # Extract episode_id from parent directory name
    ep_id = output_log.parent.name
    try:
        ep_id = int(ep_id)
    except ValueError:
        pass   # non-numeric directory name — keep it as a string, it still joins fine
    metrics["episode_id"] = ep_id
    return metrics


def _load_owmm_results(owmm_dir: Path) -> List[Dict]:
    """Walk OWMM eval directory and collect per-episode metrics.

    Missing directory is a warning, not an error: comparing against nothing still prints our
    own numbers, which is often all you want.
    """
    records = []
    if not owmm_dir.exists():
        print(f"[compare] OWMM eval dir not found: {owmm_dir}")
        return records
    for ep_dir in sorted(owmm_dir.iterdir()):
        log = ep_dir / "output.log"
        rec = _parse_owmm_log(log)
        if rec:
            rec["framework"] = "owmm"
            records.append(rec)
    return records


# ── EmbodiedAgent CSV loading ─────────────────────────────────────────────────

def _load_embodied_results(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        print(f"[compare] EmbodiedAgent CSV not found: {csv_path}")
        return []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(records: List[Dict], framework: str) -> Dict:
    """Mean of each metric across one framework's episodes.

    Because success metrics are 0/1 per episode, their mean IS the success rate — which is why
    the same _avg() serves both the rate rows and the step-count rows in the table.
    """
    recs = [r for r in records if r.get("framework") == framework or not r.get("framework")]
    if not recs:
        return {"framework": framework, "n": 0}

    def _avg(key, fallback_keys=()):
        """Mean over episodes that HAVE this metric, trying the fallback spellings.

        The fallbacks are the hab_-prefix reconciliation (see the module docstring). Values
        that are missing or unparseable are skipped rather than treated as 0 — a metric an
        episode never reported must not drag the average down.
        """
        vals = []
        for r in recs:
            v = r.get(key)
            if v is None:
                for fk in fallback_keys:
                    v = r.get(fk)
                    if v is not None:
                        break
            try:
                # CSV values arrive as strings; OWMM's arrive as floats. float() handles both.
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
        return sum(vals) / len(vals) if vals else None   # None ⇒ printed as "—"

    n = len(recs)
    # The three stages are OVMM's decomposition of the task: reach the object, pick it up,
    # reach the goal. They are the most useful rows in the table — they say WHERE a failing
    # agent is failing, which the single success rate cannot.
    return {
        "framework"         : framework,
        "n"                 : n,
        "pddl_success_rate" : _avg("pddl_success", ("hab_pddl_success",)),
        "stage1_success"    : _avg("pddl_stage_goals.stage_1_success",
                                   ("hab_pddl_stage_goals.stage_1_success",)),
        "stage2_success"    : _avg("pddl_stage_goals.stage_2_success",
                                   ("hab_pddl_stage_goals.stage_2_success",)),
        "stage3_success"    : _avg("pddl_stage_goals.stage_3_success",
                                   ("hab_pddl_stage_goals.stage_3_success",)),
        "avg_num_steps"     : _avg("num_steps", ("hab_num_steps",)),
        "obj_to_goal_dist"  : _avg("object_to_goal_distance.0",
                                   ("hab_object_to_goal_distance.0",)),
    }


# ── Pretty print ──────────────────────────────────────────────────────────────

def _fmt(val, pct=False, decimals=2):
    """Format a metric, rendering a missing value as an em-dash rather than 0.

    The distinction matters when reading the table: "—" means the framework never reported
    this metric, whereas "0.0%" means it reported it and failed every episode.
    """
    if val is None:
        return "  —  "
    if pct:
        return f"{val*100:.1f}%"
    return f"{val:.{decimals}f}"


def _print_comparison(owmm: Dict, embodied: Dict, owmm_model: str, emb_model: str):
    col_w = 22

    def row(label, owmm_val, emb_val, pct=False):
        print(f"  {label:<30}  {_fmt(owmm_val, pct):<{col_w}}  {_fmt(emb_val, pct):<{col_w}}")

    sep = "─" * (34 + col_w * 2 + 2)
    print(f"\n{sep}")
    print(f"  {'Metric':<30}  {'OWMM-Agent':<{col_w}}  {'EmbodiedAgent':<{col_w}}")
    print(f"  {'Model':<30}  {owmm_model:<{col_w}}  {emb_model:<{col_w}}")
    print(f"  {'Episodes':<30}  {owmm['n']:<{col_w}}  {embodied['n']:<{col_w}}")
    print(sep)
    row("PDDL Success Rate",   owmm["pddl_success_rate"],  embodied["pddl_success_rate"],  pct=True)
    row("Stage 1 (nav to obj)",owmm["stage1_success"],     embodied["stage1_success"],     pct=True)
    row("Stage 2 (pick obj)",  owmm["stage2_success"],     embodied["stage2_success"],     pct=True)
    row("Stage 3 (nav to goal)",owmm["stage3_success"],    embodied["stage3_success"],     pct=True)
    row("Avg Steps",           owmm["avg_num_steps"],      embodied["avg_num_steps"],      pct=False)
    row("Obj→Goal Distance (m)",owmm["obj_to_goal_dist"],  embodied["obj_to_goal_dist"],   pct=False)
    print(sep)


# ── Per-episode breakdown ─────────────────────────────────────────────────────

def _per_episode_table(owmm_recs: List[Dict], emb_recs: List[Dict]):
    """Side-by-side per-episode outcomes — an OUTER join, so episodes only one framework ran
    still appear (with "—" on the other side).

    Keyed on the id as a STRING, since OWMM's may have failed to parse as an int. The sort key
    then restores numeric ordering where it can, so ep 10 doesn't sort before ep 2.
    """
    owmm_by_ep = {str(r.get("episode_id", "")): r for r in owmm_recs}
    emb_by_ep  = {str(r.get("episode_id", "")): r for r in emb_recs}
    all_eps    = sorted(set(owmm_by_ep) | set(emb_by_ep), key=lambda x: int(x) if x.isdigit() else x)

    if not all_eps:
        return

    print("\n── Per-Episode Breakdown ────────────────────────────────────────")
    print(f"  {'ep_id':<8}  {'OWMM pddl':<12}  {'Emb pddl':<12}  {'OWMM steps':<12}  {'Emb steps':<12}")
    print("  " + "─" * 60)
    # Note the asymmetric key names on each side (pddl_success vs hab_pddl_success) — the
    # hab_-prefix difference described in the module docstring, hard-coded here.
    for ep in all_eps:
        o = owmm_by_ep.get(ep, {})
        e = emb_by_ep.get(ep, {})
        o_succ  = _fmt(o.get("pddl_success"),           pct=True) if o else "  —  "
        e_succ  = _fmt(e.get("hab_pddl_success"),        pct=True) if e else "  —  "
        o_steps = _fmt(o.get("num_steps"),               pct=False) if o else "  —  "
        e_steps = _fmt(e.get("hab_num_steps"),           pct=False) if e else "  —  "
        print(f"  {ep:<8}  {o_succ:<12}  {e_succ:<12}  {o_steps:<12}  {e_steps:<12}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compare EmbodiedAgent vs OWMM-Agent results")
    p.add_argument("--embodied",
                   default=str(_HERE / "runs" / "owmm_embodied" / "results.csv"),
                   help="Path to EmbodiedAgent results CSV")
    p.add_argument("--owmm",
                   default=str(_HERE / "OWMM-Agent" / "sim" / "habitat-lab" /
                               "eval_in_sim_info" / "sat_TEST_YCB_30scene_head_rgb"),
                   help="Directory containing OWMM-Agent episode output.log files")
    p.add_argument("--owmm_model",      default="OWMM-VLM / Gemini")
    p.add_argument("--embodied_model",  default="Gemini (EmbodiedAgent)")
    p.add_argument("--per_episode",     action="store_true",
                   help="Print per-episode breakdown table")
    args = p.parse_args()

    owmm_recs = _load_owmm_results(Path(args.owmm))
    emb_recs  = _load_embodied_results(Path(args.embodied))

    print(f"[compare] OWMM-Agent    : {len(owmm_recs)} episodes from {args.owmm}")
    print(f"[compare] EmbodiedAgent : {len(emb_recs)} episodes from {args.embodied}")

    for r in owmm_recs:
        r["framework"] = "owmm"
    for r in emb_recs:
        r["framework"] = "embodied"

    owmm_agg = _aggregate(owmm_recs, "owmm")
    emb_agg  = _aggregate(emb_recs, "embodied")

    _print_comparison(owmm_agg, emb_agg, args.owmm_model, args.embodied_model)

    if args.per_episode:
        _per_episode_table(owmm_recs, emb_recs)

    # Save merged CSV — both frameworks' raw records in one file, distinguished by the
    # `framework` column, for further analysis elsewhere.
    all_recs = owmm_recs + emb_recs
    if all_recs:
        out_csv = _HERE / "runs" / "comparison.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # Union of every key across both frameworks (they report different metric sets), so
        # no column is dropped. DictWriter fills absent keys with "".
        fieldnames = sorted({k for r in all_recs for k in r})
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_recs)
        print(f"\n[compare] merged CSV → {out_csv}")


if __name__ == "__main__":
    main()
