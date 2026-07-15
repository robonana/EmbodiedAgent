from __future__ import annotations

"""
embedding.py — Real-time FAISS index builder + in-process query.

EmbeddingWorker embeds incoming RGB frames with a frozen image extractor
(DINOv2 / SigLIP / CLIP) and maintains a FAISS IndexFlatIP written to disk
after every frame.

query_text(text, top_k)   — embed text and search the live index
query_images(paths, top_k) — embed one or more images and search

This is the recall half of the agent's episodic memory (the precision half is the VLM
rerank in toolbox_base.retrieve_memory). Everything is L2-normalised and the index is
IndexFlatIP, so the inner product FAISS maximises IS cosine similarity — that equivalence
is why normalisation appears at every point where a vector is created.

Concurrency model, which is the trickiest thing in this file:
  * Embedding runs on a background thread, so capturing a frame never blocks the sim.
  * The agent can query the index from the main thread at any time, including while the
    worker is mid-embed.
  * The extractor (a torch module) is NOT safe to use from two threads at once, and
    neither is mutating a FAISS index while searching it. `_query_lock` serialises both.
"""

import json
import os
import queue as _queue_mod
import threading

import numpy as np
import torch


class EmbeddingWorker:
    """
    Background thread that embeds frames and maintains a FAISS IndexFlatIP.

    Main thread  →  enqueue(rgb, frame_path, robot_xy, robot_yaw)
                        ↓
    Worker thread:  embed → add to FAISS  → write index.bin
                                          → write frame_paths.json
                                          → write mask_indices.json

    Main thread can also call query_text() / query_images() at any time;
    these acquire _query_lock to avoid concurrent extractor use.
    """

    # The three files that together constitute an on-disk index. FAISS stores only vectors
    # (identified by their insertion ordinal), so frame_paths.json is what maps an ordinal
    # back to the frame it came from — the index is useless without it.
    INDEX_BIN  = "index.bin"
    PATHS_JSON = "frame_paths.json"
    MIDX_JSON  = "mask_indices.json"

    def __init__(self,
                 index_dir:  str,
                 model_name: str = "siglip_base",
                 device:     str = "auto"):
        from pathlib import Path as _Path
        self.index_dir  = _Path(index_dir)
        self.model_name = model_name
        self._device    = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else device

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = str(self.index_dir / self.INDEX_BIN)
        self._paths_path = str(self.index_dir / self.PATHS_JSON)
        self._midx_path  = str(self.index_dir / self.MIDX_JSON)

        self._extractor   = None
        self._faiss_index = None   # faiss.IndexFlatIP
        # Parallel arrays indexed by FAISS ordinal. _embed_paths[i] is the frame whose
        # vector is row i of the index; they must stay in lockstep with index.ntotal.
        self._embed_paths: list[str] = []
        self._embed_midx:  list[int] = []

        self.embedded    = 0   # counters for progress reporting
        self.failed      = 0
        # Set once the extractor is loaded and any existing index restored. Queries check
        # this rather than blocking, so an early query returns [] instead of stalling the
        # agent while a multi-second model load finishes.
        self._ready      = threading.Event()
        self._query_lock = threading.Lock()   # serialises extractor access
        self._q          = _queue_mod.Queue()
        self._stop_event = threading.Event()
        # Daemon: the process must be able to exit even if the worker is mid-embed.
        self._thread     = threading.Thread(target=self._worker,
                                            daemon=True, name="EmbeddingWorker")
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────────

    def enqueue(self, rgb: np.ndarray, frame_path: str,
                robot_xy: np.ndarray, robot_yaw: float) -> None:
        """Queue a frame for embedding. Returns immediately.

        Both arrays are copied. The caller (the sim capture path) reuses its buffers, so
        without the copy the worker would embed whatever the buffer held by the time it got
        around to it — a subtle, timing-dependent corruption.
        """
        self._q.put((rgb.copy(), frame_path,
                     robot_xy.copy(), float(robot_yaw)))

    def flush(self) -> None:
        """Block until all queued frames are embedded and the index flushed.

        Relies on the worker calling task_done() on every item, including the ones it
        drops — see _worker's `finally`.
        """
        self._q.join()

    def stop(self) -> None:
        """Signal shutdown and wait for the worker to drain.

        Both the event and the None sentinel are needed: the event covers the case where
        the worker is blocked in queue.get's timeout, the sentinel wakes it immediately if
        it is waiting on an empty queue.
        """
        self._stop_event.set()
        self._q.put(None)
        self._thread.join(timeout=30.0)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def query_text(self, text: str, top_k: int = 5) -> list[dict]:
        """Embed a text string and search the live FAISS index.

        Returns list of {frame_path, score} dicts, sorted by score descending.
        Requires a model that supports text (siglip, siglip_base, clip).
        Returns [] if not ready, index empty, or model is vision-only.

        This is what makes "find the red cup" work with no detector and no labels: SigLIP/
        CLIP embed text and images into one shared space, so a text vector can be compared
        directly against frame vectors. DINOv2 has no text tower, hence the vision-only path.
        """
        if not self._ready.is_set():
            print("[EmbeddingWorker] query_text: extractor not ready.")
            return []
        with self._query_lock:   # keep the extractor and index away from the worker thread
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                print("[EmbeddingWorker] query_text: empty index.")
                return []
            try:
                import torch.nn.functional as _F
                try:
                    feat = self._extractor._encode_text([text])
                except NotImplementedError:
                    # DINOv2 and friends: no text tower. Not an error — just unsupported.
                    print(f"[EmbeddingWorker] {self.model_name} is vision-only; "
                          f"text queries not supported.")
                    return []
                # Normalise so the index's inner product is cosine similarity.
                q_vec = _F.normalize(feat, p=2, dim=-1).cpu().float().numpy()
                return self._search(q_vec, top_k)
            except Exception as e:
                # Retrieval failure degrades the agent; it must not crash it.
                print(f"[EmbeddingWorker] query_text error: {e}")
                return []

    def query_images(self, paths: list[str], top_k: int = 5) -> list[dict]:
        """Embed image(s), average, and search the live FAISS index.

        Returns list of {frame_path, score} dicts, sorted by score descending.
        Returns [] if not ready or index empty.

        Multiple query images are averaged into a single centroid vector, which asks "find
        frames like these in general" — useful for querying with several views of the same
        object. Works with every model, including the vision-only ones.
        """
        if not self._ready.is_set():
            print("[EmbeddingWorker] query_images: extractor not ready.")
            return []
        with self._query_lock:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                print("[EmbeddingWorker] query_images: empty index.")
                return []
            try:
                import torch.nn.functional as _F
                from PIL import Image as _PILImage
                vecs = []
                for p in paths:
                    if not os.path.exists(p):
                        continue   # skip missing files; any surviving image still queries
                    img = _PILImage.open(p).convert("RGB")
                    # is_query=True: some extractors apply different preprocessing to
                    # queries than to indexed frames.
                    out = self._extractor.extract({"image": img, "img_path": p},
                                                  is_query=True)
                    feat = out["keypoints"]
                    if feat.ndim == 1:
                        feat = feat.unsqueeze(0)   # (D,) -> (1, D)
                    # feat[:1] — take only the global/CLS vector even if the extractor
                    # returned per-patch features.
                    vecs.append(
                        _F.normalize(feat[:1], p=2, dim=-1).cpu().float().numpy()
                    )
                if not vecs:
                    return []
                # Mean of unit vectors is not itself a unit vector, so re-normalise —
                # otherwise the inner product would no longer be cosine similarity.
                q_vec = np.mean(vecs, axis=0)
                norm = float(np.linalg.norm(q_vec))
                if norm > 0:
                    q_vec = q_vec / norm
                return self._search(q_vec, top_k)
            except Exception as e:
                print(f"[EmbeddingWorker] query_images error: {e}")
                return []

    # ── internals ─────────────────────────────────────────────────────────────

    def _search(self, q_vec: np.ndarray, top_k: int) -> list[dict]:
        """Run FAISS search. Caller must hold _query_lock.

        top_k == -1 means "return everything" (used when a time filter will be applied
        afterwards). Otherwise clamp to ntotal — FAISS pads with -1 indices when asked for
        more neighbours than it holds, and the bounds check below would drop them anyway.
        """
        k = (self._faiss_index.ntotal if top_k == -1
             else min(top_k, self._faiss_index.ntotal))
        scores, indices = self._faiss_index.search(q_vec, k)
        results = []
        # [0] — one query vector, so one row of results.
        for idx, score in zip(indices[0], scores[0]):
            i = int(idx)
            # Guards against FAISS's -1 padding and against any drift between the index
            # and the parallel path array.
            if 0 <= i < len(self._embed_paths):
                results.append({"frame_path": self._embed_paths[i],
                                 "score": float(score)})
        return results

    def _load_extractor(self) -> bool:
        """Load self-contained extractor from memory.extractors.

        Returns False rather than raising: a machine without the model weights should run
        the agent with retrieval disabled, not fail to start.
        """
        try:
            from .extractors import make_extractor, MODELS
            self._extractor = make_extractor(self.model_name, self._device)
            self._vec_dim   = MODELS[self.model_name][1]
            print(f"[EmbeddingWorker] extractor ready: {self.model_name} "
                  f"({self._vec_dim}-d) on {self._device}")
            return True
        except Exception as e:
            print(f"[EmbeddingWorker] extractor load failed: {e}")
            return False

    def _load_existing_index(self) -> None:
        """Warm-start from an on-disk index if present.

        This is what lets an episode reuse the pre-episode scene scan instead of re-embedding
        every frame on each run. Both the index and the path list must be present — an index
        without its paths is unusable, since ordinals could not be mapped back to frames.

        On any failure we reset all three structures and start fresh. Partially restored
        state would be worse than none: it would silently misattribute vectors to the wrong
        frames.
        """
        try:
            import faiss as _faiss
            if (os.path.exists(self._index_path) and
                    os.path.exists(self._paths_path)):
                self._faiss_index = _faiss.read_index(self._index_path)
                with open(self._paths_path) as f:
                    self._embed_paths = json.load(f)
                if os.path.exists(self._midx_path):
                    with open(self._midx_path) as f:
                        self._embed_midx = json.load(f)
                else:
                    # Older index without the mask sidecar — synthesise zeros so the
                    # parallel arrays stay the same length.
                    self._embed_midx = [0] * len(self._embed_paths)
                print(f"[EmbeddingWorker] warm-started: "
                      f"{self._faiss_index.ntotal} vectors from disk")
        except Exception as e:
            print(f"[EmbeddingWorker] warm-start failed ({e}), starting fresh.")
            self._faiss_index = None
            self._embed_paths = []
            self._embed_midx  = []

    def _embed(self, rgb: np.ndarray, frame_path: str) -> "np.ndarray | None":
        """Return (1, D) L2-normalised float32 embedding or None on error.

        float32 because that is FAISS's storage type; detach() because we are doing
        inference and have no use for the autograd graph.
        """
        from PIL import Image as _PILImage
        import torch.nn.functional as _F
        try:
            img = _PILImage.fromarray(rgb).convert("RGB")
            # is_query=False — this frame is going *into* the index, not querying it.
            out  = self._extractor.extract(
                {"image": img, "img_path": frame_path}, is_query=False
            )
            feat = out["keypoints"]
            if feat.ndim == 1:
                feat = feat.unsqueeze(0)
            feat = _F.normalize(feat[:1], p=2, dim=-1)
            return feat.cpu().float().detach().numpy()
        except Exception as e:
            print(f"[EmbeddingWorker] embed error for "
                  f"{os.path.basename(frame_path)}: {e}")
            return None

    def _save_index(self) -> None:
        """Persist all three files. Called after every single frame.

        Per-frame writes are wasteful in principle but right in practice: a scan can be
        interrupted at any moment, and re-embedding a whole scene is far more expensive
        than these writes. It also means a separate process can read a usable index while
        the scan is still running.
        """
        import faiss as _faiss
        try:
            _faiss.write_index(self._faiss_index, self._index_path)
            with open(self._paths_path, "w") as f:
                json.dump(self._embed_paths, f)
            with open(self._midx_path, "w") as f:
                json.dump(self._embed_midx, f)
        except Exception as e:
            print(f"[EmbeddingWorker] index save error: {e}")

    def _worker(self) -> None:
        """Background thread body: load the model, restore the index, then consume frames."""
        import faiss as _faiss

        with self._query_lock:
            loaded = self._load_extractor()

        if not loaded:
            # Degenerate mode: no extractor, so nothing can be embedded. We still have to
            # drain the queue — otherwise every enqueue() would accumulate forever and any
            # flush() would block for good. Note _ready is never set, so queries return [].
            print("[EmbeddingWorker] disabled — extractor unavailable.")
            while True:
                item = self._q.get()
                self._q.task_done()
                if item is None:
                    break
            return

        with self._query_lock:
            self._load_existing_index()

        # Only now can queries succeed.
        self._ready.set()

        while True:
            try:
                # Timeout rather than a blocking get, so a stop() that arrives while the
                # queue is empty is noticed within a second.
                item = self._q.get(timeout=1.0)
            except _queue_mod.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if item is None:   # shutdown sentinel from stop()
                self._q.task_done()
                break

            rgb, frame_path, robot_xy, robot_yaw = item
            try:
                with self._query_lock:
                    vec = self._embed(rgb, frame_path)
                    if vec is not None:
                        # Lazily create the index on the first vector — its dimensionality
                        # comes from the extractor's actual output, not a hard-coded constant.
                        if self._faiss_index is None:
                            self._faiss_index = _faiss.IndexFlatIP(vec.shape[1])
                        # These three appends must stay in lockstep: FAISS row i ↔
                        # _embed_paths[i] ↔ _embed_midx[i].
                        self._faiss_index.add(vec)
                        self._embed_paths.append(frame_path)
                        self._embed_midx.append(0)
                        self._save_index()
                        self.embedded += 1
                        print(f"[EmbeddingWorker] #{self.embedded:04d}  "
                              f"{os.path.basename(frame_path)}  "
                              f"index={self._faiss_index.ntotal} vectors")
                    else:
                        self.failed += 1   # a bad frame is skipped, not fatal
            except Exception as e:
                print(f"[EmbeddingWorker] processing error: {e}")
                self.failed += 1
            finally:
                # Must run on every path (success, skip, or exception), or flush() hangs.
                self._q.task_done()
