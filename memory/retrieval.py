"""
retrieval.py — Memory retrieval for the agent pipeline.

retrieve_memory_candidates   Search the FAISS index via EmbeddingWorker and
                             return MemoryCandidate objects for the agent.
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
    """
    if embedding_worker is None:
        print("[retrieve_memory_candidates] no embedding_worker provided.")
        return []
    if not embedding_worker.is_ready:
        print("[retrieve_memory_candidates] embedding_worker not ready yet.")
        return []

    fetch_k = embedding_worker._faiss_index.ntotal if top_k == -1 else top_k
    if fetch_k == 0:
        return []

    print(f"[retrieve_memory_candidates] query={query!r}  "
          f"model={model}  top_k={'all' if top_k == -1 else top_k}")

    if query_image_paths:
        raw_frames = embedding_worker.query_images(query_image_paths, top_k=fetch_k)
    else:
        raw_frames = embedding_worker.query_text(query, top_k=fetch_k)

    if not raw_frames:
        return []

    from agent.schemas import MemoryCandidate

    candidates = []
    for frame in raw_frames:
        frame_path = frame.get("frame_path", "")
        score      = float(frame.get("score", 0.0))

        stem = os.path.splitext(os.path.basename(frame_path))[0]
        try:
            frame_idx = int(stem)
        except ValueError:
            frame_idx = None
        memory_id = (f"mem_{frame_idx:06d}" if frame_idx is not None
                     else f"mem_{stem}")

        pose: list[float] = []
        if frame_idx is not None:
            xy_path = os.path.join(
                capture_out_dir, "robot_xy", f"{frame_idx:06d}.txt")
            if os.path.exists(xy_path):
                try:
                    import numpy as _np
                    data = _np.loadtxt(xy_path).flatten()
                    pose = [float(data[0]), float(data[1]),
                            float(data[2]) if len(data) >= 3 else 0.0]
                except Exception:
                    pass

        candidates.append(MemoryCandidate(
            memory_id=memory_id,
            image_path=frame_path,
            robot_pose=pose,
            retrieval_score=score,
            frame_idx=frame_idx,
        ))

    if episodic_memory is not None:
        try:
            episodic_memory.enrich_candidates(candidates)
        except Exception as e:
            print(f"[retrieve_memory_candidates] enrichment error: {e}")

    print(f"[retrieve_memory_candidates] returning {len(candidates)} candidates")
    return candidates
