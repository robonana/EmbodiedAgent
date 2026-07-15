"""
retrieval.py — Memory retrieval for the agent pipeline.

retrieve_memory_candidates   Search the FAISS index via EmbeddingWorker and
                             return MemoryCandidate objects for the agent.

The glue layer between the embedding index and the agent. EmbeddingWorker speaks in
{frame_path, score}; the agent speaks in MemoryCandidate (memory_id, pose, timestamp).
This module performs that join, pulling each frame's pose off disk and its timestamp out
of EpisodicMemory.
"""

import os
from typing import List, Optional


def retrieve_memory_candidates(
    query: str,
    index_dir: str,
    capture_out_dir: str,
    top_k: int = 5,
    model: str = "siglip_base",
    retrieval_data_root: Optional[str] = None,
    scene_id: Optional[str] = None,
    episodic_memory=None,
    embedding_worker=None,
    query_image_paths: Optional[List[str]] = None,
) -> list:
    """
    Search the FAISS index and return top-k MemoryCandidate objects.

    Queries the live EmbeddingWorker index directly (no subprocess).
    Returns [] if the worker is not ready or the index is empty.

    query             : text description (used for text-based retrieval)
    query_image_paths : optional image paths (used instead of text if provided)
    embedding_worker  : EmbeddingWorker instance (required for querying)

    "Live" is the key word: this searches the in-process index that the worker is still
    appending to, so frames the robot captured seconds ago are already retrievable. Earlier
    versions shelled out to a separate indexing process; this does not.

    Returns [] on every failure path rather than raising — the caller (toolbox.retrieve_memory)
    turns an empty list into an ok=False ToolResult that tells the model to look elsewhere.
    """
    if embedding_worker is None:
        print("[retrieve_memory_candidates] no embedding_worker provided.")
        return []
    if not embedding_worker.is_ready:
        # Model still loading. Common on the first step of a run; not an error.
        print("[retrieve_memory_candidates] embedding_worker not ready yet.")
        return []

    # -1 means "everything in the index" (the caller then applies a time filter).
    fetch_k = embedding_worker._faiss_index.ntotal if top_k == -1 else top_k
    if fetch_k == 0:
        return []   # nothing indexed yet

    print(f"[retrieve_memory_candidates] query={query!r}  "
          f"model={model}  top_k={'all' if top_k == -1 else top_k}")

    # Image query wins when both are present: the caller only sets query_image_paths when
    # `query` was itself a file path, so the text form would be a meaningless filename.
    if query_image_paths:
        raw_frames = embedding_worker.query_images(query_image_paths, top_k=fetch_k)
    else:
        raw_frames = embedding_worker.query_text(query, top_k=fetch_k)

    if not raw_frames:
        return []

    # Imported here, not at module scope, to keep memory/ from importing agent/ at load
    # time (the dependency runs the other way round everywhere else).
    from agent.schemas import MemoryCandidate

    candidates = []
    for frame in raw_frames:
        frame_path = frame.get("frame_path", "")
        score      = float(frame.get("score", 0.0))

        # Recover the frame index from the filename ("000042.png" → 42). This filename ⇄
        # memory_id convention is relied on throughout the system, and is documented to the
        # VLM in the system prompt so it can cite a memory_id for any frame it has seen.
        stem = os.path.splitext(os.path.basename(frame_path))[0]
        try:
            frame_idx = int(stem)
        except ValueError:
            frame_idx = None   # non-numeric filename — still usable, just not pose-linked
        memory_id = (f"mem_{frame_idx:06d}" if frame_idx is not None
                     else f"mem_{stem}")

        # Join in the pose recorded when the frame was captured. Without this the candidate
        # cannot be a navigate() target — pose is what makes a memory *actionable* rather
        # than merely informative.
        pose: list[float] = []
        if frame_idx is not None:
            xy_path = os.path.join(
                capture_out_dir, "robot_xy", f"{frame_idx:06d}.txt")
            if os.path.exists(xy_path):
                try:
                    import numpy as _np
                    data = _np.loadtxt(xy_path).flatten()
                    # Files hold [x, y] or [x, y, yaw]; default a missing yaw to 0.
                    pose = [float(data[0]), float(data[1]),
                            float(data[2]) if len(data) >= 3 else 0.0]
                except Exception:
                    pass   # keep the candidate; it just won't be navigable

        candidates.append(MemoryCandidate(
            memory_id=memory_id,
            image_path=frame_path,
            robot_pose=pose,
            retrieval_score=score,
            frame_idx=frame_idx,
        ))

    # Second join: timestamps from the record store, needed for the time-window filter and
    # shown to the model in the rerank prompt. Best-effort — a failed enrichment costs
    # timestamps, not candidates.
    if episodic_memory is not None:
        try:
            episodic_memory.enrich_candidates(candidates)
        except Exception as e:
            print(f"[retrieve_memory_candidates] enrichment error: {e}")

    print(f"[retrieve_memory_candidates] returning {len(candidates)} candidates")
    return candidates
