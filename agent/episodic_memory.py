"""
agent/episodic_memory.py — Robot-observation-only episodic memory store.

EpisodicMemory persists MemoryEntry records built exclusively from sensor data
(RGB image path + robot pose).  Never stores simulator ground-truth object names,
poses, or segmentation.

Storage layout:
    {memory_dir}/
        entries.jsonl       append-only, one MemoryEntry JSON per line
        memory_index.json   {"memory_ids": [...], "id_to_line": {...}}

Why two files: entries.jsonl is the durable append-only log (cheap writes, survives a
crash mid-episode), while memory_index.json is a derived lookup table rewritten on every
append so a fresh process can resolve a memory_id to a line number without scanning the
whole log. The index is disposable — it could always be rebuilt from the JSONL.

This class stores and looks up; it does *not* do similarity search. Retrieval is the
FAISS index owned by EmbeddingWorker (see memory/embedding.py), which returns memory_ids
that are then resolved back to full records here.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .schemas import (
    EmbeddingRefs,
    MemoryCandidate,
    MemoryEntry,
    MemorySource,
    SensorData,
)


def frame_to_memory_id(frame_idx: int) -> str:
    """Stable memory ID for a captured frame.  e.g. 42 → 'mem_000042'.

    Deterministic from the frame index (rather than, say, a UUID) so that re-importing
    the same scan directory is idempotent — the ids collide with what's already indexed
    and get skipped instead of duplicated.
    """
    return f"mem_{frame_idx:06d}"


class EpisodicMemory:
    """
    Append-only episodic memory store for robot observations.

    Thread-safe: add_entry() and get_entry() may be called from the
    main thread (agent loop) and background threads (SensoryTick, scan loop).

    Retrieval delegates to the existing FAISS index maintained by EmbeddingWorker;
    this class only provides persistent storage and fast lookup by memory_id.
    """

    _ENTRIES_FILE = "entries.jsonl"
    _INDEX_FILE   = "memory_index.json"

    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self._entries_path = self.memory_dir / self._ENTRIES_FILE
        self._index_path   = self.memory_dir / self._INDEX_FILE

        # Guards _id_to_line / _ordered_ids *and* the append to entries.jsonl — the two
        # must stay consistent, since a line number is only meaningful if the line was
        # actually written at that offset.
        self._lock    = threading.Lock()
        # id → line number (0-based) in entries.jsonl
        self._id_to_line: dict[str, int] = {}
        # Insertion order, which is also line order — len() of this is the next line no.
        self._ordered_ids: list[str] = []

        self._load_index()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_entry(self, entry: MemoryEntry) -> None:
        """Append a MemoryEntry to the JSONL store and update the index.

        Everything happens under the lock: the line number is derived from the current
        length, so a concurrent append between "compute line_no" and "write the line"
        would corrupt the mapping.
        """
        with self._lock:
            line_no = len(self._ordered_ids)
            with open(self._entries_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
            self._id_to_line[entry.memory_id] = line_no
            self._ordered_ids.append(entry.memory_id)
            self._save_index_locked()

    def get_entry(self, memory_id: str) -> Optional[MemoryEntry]:
        """Look up a MemoryEntry by its memory_id.  Returns None if not found.

        The lock is held only for the (fast) dict lookup, then released before the file
        read — the log is append-only, so line `line_no` can never be moved or rewritten
        by a concurrent writer, and reading it unlocked is safe.

        The read itself is a linear scan to the target line rather than a byte-offset
        seek. Memories number in the thousands and lookups are rare (only when the VLM
        cites an id), so the simplicity is worth more than the O(1) seek would be.
        """
        with self._lock:
            line_no = self._id_to_line.get(memory_id)
            if line_no is None:
                return None
        try:
            with open(self._entries_path) as f:
                for i, line in enumerate(f):
                    if i == line_no:
                        return MemoryEntry.from_dict(json.loads(line))
        except Exception as e:
            # Never raise into the agent loop over a memory miss — a None here just
            # becomes "memory not found" feedback to the VLM.
            print(f"[EpisodicMemory] get_entry({memory_id}) error: {e}")
        return None

    def get_pose(self, memory_id: str) -> Optional[tuple[np.ndarray, Optional[float]]]:
        """
        Return (xy, yaw) for a memory_id.
        xy is np.ndarray([x, y]).  yaw may be None if not stored.

        This is the bridge that makes `navigate` work: the VLM can only name a memory_id,
        and this turns that id back into the metric goal the planner needs. Yaw is
        optional because some capture paths only record position.
        """
        entry = self.get_entry(memory_id)
        if entry is None:
            return None
        pose = entry.sensor.robot_pose
        if len(pose) >= 2:
            xy  = np.array([float(pose[0]), float(pose[1])])
            yaw = float(pose[2]) if len(pose) >= 3 else None
            return xy, yaw
        return None  # pose was empty/degenerate — treat as "no pose known"

    def create_entry(
        self,
        memory_id: str,
        image_path: str,
        robot_pose: list[float],
        timestamp: Optional[str] = None,
        embedding_model: str = "unknown",
        source_type: str = "agent_observe",
        episode_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Construct a MemoryEntry without persisting it.  Call add_entry() to save.

        Split from add_entry() so a caller can build the record, hand it to the embedding
        worker to fill in EmbeddingRefs, and only then persist it.
        """
        return MemoryEntry(
            memory_id=memory_id,
            sensor=SensorData(
                image_path=image_path,
                robot_pose=robot_pose,
                timestamp=timestamp or time.strftime("%H:%M:%S"),
            ),
            # image_embedding_path is left None here; the EmbeddingWorker fills it in
            # once the vector has actually been computed and written.
            embeddings=EmbeddingRefs(embedding_model=embedding_model),
            source=MemorySource(
                source_type=source_type,
                episode_id=episode_id,
            ),
        )

    def enrich_candidates(
        self, candidates: list[MemoryCandidate]
    ) -> list[MemoryCandidate]:
        """
        Enrich each MemoryCandidate in-place with sensor data from EpisodicMemory.

        Currently fills timestamp from the stored MemoryEntry.
        Returns the same list for chaining.

        Needed because FAISS returns only (id, score) — everything the prompt wants to
        *show* about a hit has to be joined back in from the record store.
        """
        for c in candidates:
            entry = self.get_entry(c.memory_id)
            if entry is None:
                continue  # stale index entry; skip rather than fail the whole retrieval
            if c.timestamp is None and entry.sensor.timestamp:
                c.timestamp = entry.sensor.timestamp
        return candidates

    def build_from_scan_dir(
        self,
        capture_out_dir: str,
        embedding_model: str = "unknown",
    ) -> int:
        """
        Import existing scan frames into EpisodicMemory.

        Reads color/{idx:06d}.png + robot_xy/{idx:06d}.txt pairs.
        Skips frames already present in the index.  Returns count of new entries.

        This is how the pre-episode teleoperated scene scan becomes queryable memory:
        the scan writes plain PNG + pose-text pairs, and this folds them into the same
        store the agent's own observations go into (tagged source_type="scan_wasd").
        Idempotent, so it is safe to call on every run against a scan dir that grows.
        """
        color_dir = os.path.join(capture_out_dir, "color")
        xy_dir    = os.path.join(capture_out_dir, "robot_xy")
        if not os.path.isdir(color_dir):
            return 0  # no scan was performed for this scene

        imported = 0
        # Sorted so memory ids are assigned in frame order, matching capture order.
        for fname in sorted(os.listdir(color_dir)):
            if not fname.endswith(".png"):
                continue
            stem = os.path.splitext(fname)[0]
            try:
                frame_idx = int(stem)
            except ValueError:
                continue  # not a frame file (e.g. a stray thumbnail) — ignore

            memory_id = frame_to_memory_id(frame_idx)
            if memory_id in self._id_to_line:
                continue  # already imported on a previous run

            image_path = os.path.join(color_dir, fname)

            # The pose sidecar is optional: a frame with no pose still carries useful
            # visual information for retrieval, it just can't be a `navigate` target.
            pose: list[float] = []
            xy_path = os.path.join(xy_dir, f"{frame_idx:06d}.txt")
            if os.path.exists(xy_path):
                try:
                    data = np.loadtxt(xy_path).flatten()
                    # Files may hold [x, y] or [x, y, yaw]; default the missing yaw to 0.
                    pose = [float(data[0]), float(data[1]),
                            float(data[2]) if len(data) >= 3 else 0.0]
                except Exception:
                    pass

            entry = self.create_entry(
                memory_id=memory_id,
                image_path=image_path,
                robot_pose=pose,
                embedding_model=embedding_model,
                source_type="scan_wasd",   # provenance: teleop scan, not agent action
            )
            self.add_entry(entry)
            imported += 1

        if imported:
            print(f"[EpisodicMemory] imported {imported} scan frames from {color_dir}")
        return imported

    def __len__(self) -> int:
        return len(self._ordered_ids)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_index(self) -> None:
        """Restore the id→line map from disk. Absent or corrupt index ⇒ start empty.

        Note this makes a corrupt index silently shadow an existing entries.jsonl (the
        log is still there, but nothing points into it). Recovery is to delete
        memory_index.json and re-import.
        """
        if not self._index_path.exists():
            return
        try:
            with open(self._index_path) as f:
                idx = json.load(f)
            self._id_to_line   = idx.get("id_to_line", {})
            self._ordered_ids  = idx.get("memory_ids", [])
        except Exception as e:
            print(f"[EpisodicMemory] could not load index: {e}")

    def _save_index_locked(self) -> None:
        """Must be called with self._lock held.

        Rewrites the whole index on every append. That's O(n) per entry, but n is small
        (thousands) and the write keeps the on-disk index consistent with the log at all
        times, which is worth far more than the write cost during an episode.
        """
        try:
            with open(self._index_path, "w") as f:
                json.dump({
                    "memory_ids": self._ordered_ids,
                    "id_to_line": self._id_to_line,
                }, f)
        except Exception as e:
            print(f"[EpisodicMemory] index save error: {e}")
