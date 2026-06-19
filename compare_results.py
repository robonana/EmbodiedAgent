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
    """Parse a single OWMM-Agent episode output.log → metric dict."""
    if not output_log.exists():
        return None
    text = output_log.read_text(errors="replace")
    metrics = {}
    for line in text.splitlines():
        m = re.search(r"Average episode (\S+):\s+([\d.]+)", line)
        if m:
            metrics[m.group(1)] = float(m.group(2))
    if not metrics:
        return None
    # Extract episode_id from parent directory name
    ep_id = output_log.parent.name
    try:
        ep_id = int(ep_id)
    except ValueError:
        pass
    metrics["episode_id"] = ep_id
    return metrics


def _load_owmm_results(owmm_dir: Path) -> List[Dict]:
    """Walk OWMM eval directory and collect per-episode metrics."""
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
    recs = [r for r in records if r.get("framework") == framework or not r.get("framework")]
    if not recs:
        return {"framework": framework, "n": 0}

    def _avg(key, fallback_keys=()):
        vals = []
        for r in recs:
            v = r.get(key)
            if v is None:
                for fk in fallback_keys:
                    v = r.get(fk)
                    if v is not None:
                        break
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
        return sum(vals) / len(vals) if vals else None

    n = len(recs)
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
    owmm_by_ep = {str(r.get("episode_id", "")): r for r in owmm_recs}
    emb_by_ep  = {str(r.get("episode_id", "")): r for r in emb_recs}
    all_eps    = sorted(set(owmm_by_ep) | set(emb_by_ep), key=lambda x: int(x) if x.isdigit() else x)

    if not all_eps:
        return

    print("\n── Per-Episode Breakdown ────────────────────────────────────────")
    print(f"  {'ep_id':<8}  {'OWMM pddl':<12}  {'Emb pddl':<12}  {'OWMM steps':<12}  {'Emb steps':<12}")
    print("  " + "─" * 60)
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

    # Save merged CSV
    all_recs = owmm_recs + emb_recs
    if all_recs:
        out_csv = _HERE / "runs" / "comparison.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for r in all_recs for k in r})
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_recs)
        print(f"\n[compare] merged CSV → {out_csv}")


if __name__ == "__main__":
    main()
