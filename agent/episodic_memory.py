"""
agent/episodic_memory.py — Robot-observation-only episodic memory store.

EpisodicMemory persists MemoryEntry records built exclusively from sensor data
(RGB image path + robot pose).  Never stores simulator ground-truth object names,
poses, or segmentation.

Storage layout:
    {memory_dir}/
        entries.jsonl       append-only, one MemoryEntry JSON per line
        memory_index.json   {"memory_ids": [...], "id_to_line": {...}}
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
    """Stable memory ID for a captured frame.  e.g. 42 → 'mem_000042'."""
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

        self._lock    = threading.Lock()
        # id → line number (0-based) in entries.jsonl
        self._id_to_line: dict[str, int] = {}
        self._ordered_ids: list[str] = []

        self._load_index()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_entry(self, entry: MemoryEntry) -> None:
        """Append a MemoryEntry to the JSONL store and update the index."""
        with self._lock:
            line_no = len(self._ordered_ids)
            with open(self._entries_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
            self._id_to_line[entry.memory_id] = line_no
            self._ordered_ids.append(entry.memory_id)
            self._save_index_locked()

    def get_entry(self, memory_id: str) -> Optional[MemoryEntry]:
        """Look up a MemoryEntry by its memory_id.  Returns None if not found."""
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
            print(f"[EpisodicMemory] get_entry({memory_id}) error: {e}")
        return None

    def get_pose(self, memory_id: str) -> Optional[tuple[np.ndarray, Optional[float]]]:
        """
        Return (xy, yaw) for a memory_id.
        xy is np.ndarray([x, y]).  yaw may be None if not stored.
        """
        entry = self.get_entry(memory_id)
        if entry is None:
            return None
        pose = entry.sensor.robot_pose
        if len(pose) >= 2:
            xy  = np.array([float(pose[0]), float(pose[1])])
            yaw = float(pose[2]) if len(pose) >= 3 else None
            return xy, yaw
        return None

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
        """Construct a MemoryEntry without persisting it.  Call add_entry() to save."""
        return MemoryEntry(
            memory_id=memory_id,
            sensor=SensorData(
                image_path=image_path,
                robot_pose=robot_pose,
                timestamp=timestamp or time.strftime("%H:%M:%S"),
            ),
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
        """
        for c in candidates:
            entry = self.get_entry(c.memory_id)
            if entry is None:
                continue
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
        """
        color_dir = os.path.join(capture_out_dir, "color")
        xy_dir    = os.path.join(capture_out_dir, "robot_xy")
        if not os.path.isdir(color_dir):
            return 0

        imported = 0
        for fname in sorted(os.listdir(color_dir)):
            if not fname.endswith(".png"):
                continue
            stem = os.path.splitext(fname)[0]
            try:
                frame_idx = int(stem)
            except ValueError:
                continue

            memory_id = frame_to_memory_id(frame_idx)
            if memory_id in self._id_to_line:
                continue

            image_path = os.path.join(color_dir, fname)
            pose: list[float] = []
            xy_path = os.path.join(xy_dir, f"{frame_idx:06d}.txt")
            if os.path.exists(xy_path):
                try:
                    data = np.loadtxt(xy_path).flatten()
                    pose = [float(data[0]), float(data[1]),
                            float(data[2]) if len(data) >= 3 else 0.0]
                except Exception:
                    pass

            entry = self.create_entry(
                memory_id=memory_id,
                image_path=image_path,
                robot_pose=pose,
                embedding_model=embedding_model,
                source_type="scan_wasd",
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
        """Must be called with self._lock held."""
        try:
            with open(self._index_path, "w") as f:
                json.dump({
                    "memory_ids": self._ordered_ids,
                    "id_to_line": self._id_to_line,
                }, f)
        except Exception as e:
            print(f"[EpisodicMemory] index save error: {e}")
