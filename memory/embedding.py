"""
embedding.py — Real-time FAISS index builder + in-process query.

EmbeddingWorker embeds incoming RGB frames with a frozen image extractor
(DINOv2 / SigLIP / CLIP) and maintains a FAISS IndexFlatIP written to disk
after every frame.

query_text(text, top_k)   — embed text and search the live index
query_images(paths, top_k) — embed one or more images and search
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
        self._embed_paths: list[str] = []
        self._embed_midx:  list[int] = []

        self.embedded    = 0
        self.failed      = 0
        self._ready      = threading.Event()
        self._query_lock = threading.Lock()   # serialises extractor access
        self._q          = _queue_mod.Queue()
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._worker,
                                            daemon=True, name="EmbeddingWorker")
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────────

    def enqueue(self, rgb: np.ndarray, frame_path: str,
                robot_xy: np.ndarray, robot_yaw: float) -> None:
        """Queue a frame for embedding. Returns immediately."""
        self._q.put((rgb.copy(), frame_path,
                     robot_xy.copy(), float(robot_yaw)))

    def flush(self) -> None:
        """Block until all queued frames are embedded and the index flushed."""
        self._q.join()

    def stop(self) -> None:
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
        """
        if not self._ready.is_set():
            print("[EmbeddingWorker] query_text: extractor not ready.")
            return []
        with self._query_lock:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                print("[EmbeddingWorker] query_text: empty index.")
                return []
            try:
                import torch.nn.functional as _F
                try:
                    feat = self._extractor._encode_text([text])
                except NotImplementedError:
                    print(f"[EmbeddingWorker] {self.model_name} is vision-only; "
                          f"text queries not supported.")
                    return []
                q_vec = _F.normalize(feat, p=2, dim=-1).cpu().float().numpy()
                return self._search(q_vec, top_k)
            except Exception as e:
                print(f"[EmbeddingWorker] query_text error: {e}")
                return []

    def query_images(self, paths: list[str], top_k: int = 5) -> list[dict]:
        """Embed image(s), average, and search the live FAISS index.

        Returns list of {frame_path, score} dicts, sorted by score descending.
        Returns [] if not ready or index empty.
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
                        continue
                    img = _PILImage.open(p).convert("RGB")
                    out = self._extractor.extract({"image": img, "img_path": p},
                                                  is_query=True)
                    feat = out["keypoints"]
                    if feat.ndim == 1:
                        feat = feat.unsqueeze(0)
                    vecs.append(
                        _F.normalize(feat[:1], p=2, dim=-1).cpu().float().numpy()
                    )
                if not vecs:
                    return []
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
        """Run FAISS search. Caller must hold _query_lock."""
        k = (self._faiss_index.ntotal if top_k == -1
             else min(top_k, self._faiss_index.ntotal))
        scores, indices = self._faiss_index.search(q_vec, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            i = int(idx)
            if 0 <= i < len(self._embed_paths):
                results.append({"frame_path": self._embed_paths[i],
                                 "score": float(score)})
        return results

    def _load_extractor(self) -> bool:
        """Load self-contained extractor from memory.extractors."""
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
        """Warm-start from an on-disk index if present."""
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
                    self._embed_midx = [0] * len(self._embed_paths)
                print(f"[EmbeddingWorker] warm-started: "
                      f"{self._faiss_index.ntotal} vectors from disk")
        except Exception as e:
            print(f"[EmbeddingWorker] warm-start failed ({e}), starting fresh.")
            self._faiss_index = None
            self._embed_paths = []
            self._embed_midx  = []

    def _embed(self, rgb: np.ndarray, frame_path: str) -> "np.ndarray | None":
        """Return (1, D) L2-normalised float32 embedding or None on error."""
        from PIL import Image as _PILImage
        import torch.nn.functional as _F
        try:
            img = _PILImage.fromarray(rgb).convert("RGB")
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
        import faiss as _faiss

        with self._query_lock:
            loaded = self._load_extractor()

        if not loaded:
            print("[EmbeddingWorker] disabled — extractor unavailable.")
            while True:
                item = self._q.get()
                self._q.task_done()
                if item is None:
                    break
            return

        with self._query_lock:
            self._load_existing_index()

        self._ready.set()

        while True:
            try:
                item = self._q.get(timeout=1.0)
            except _queue_mod.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if item is None:
                self._q.task_done()
                break

            rgb, frame_path, robot_xy, robot_yaw = item
            try:
                with self._query_lock:
                    vec = self._embed(rgb, frame_path)
                    if vec is not None:
                        if self._faiss_index is None:
                            self._faiss_index = _faiss.IndexFlatIP(vec.shape[1])
                        self._faiss_index.add(vec)
                        self._embed_paths.append(frame_path)
                        self._embed_midx.append(0)
                        self._save_index()
                        self.embedded += 1
                        print(f"[EmbeddingWorker] #{self.embedded:04d}  "
                              f"{os.path.basename(frame_path)}  "
                              f"index={self._faiss_index.ntotal} vectors")
                    else:
                        self.failed += 1
            except Exception as e:
                print(f"[EmbeddingWorker] processing error: {e}")
                self.failed += 1
            finally:
                self._q.task_done()
