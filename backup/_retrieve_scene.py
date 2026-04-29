"""
Retrieve scene frames that contain a query object using image embeddings.

Subcommands:
    build   — (optionally) generate SAM masks, extract embeddings, save FAISS index.
    query   — Search the index with one or more query images and display ranked results.

Usage:
    python retrieve_scene.py build --scene SCENE_ID [OPTIONS]
    python retrieve_scene.py query --scene SCENE_ID --query IMAGE [OPTIONS]

Models (--model):
    dinov2        DINOv2 Large  1024-d  (default)
    dinov2_base   DINOv2 Base    768-d
    clip          CLIP ViT-B/16  512-d
    siglip        SigLIP so400m 1152-d
    siglip_base   SigLIP Base    768-d

With --with_masks (recommended):
    Runs OWLv2 object detection + SAM segmentation on scene frames first, then
    uses Mask Inversion (Stage B) to produce object-aware embeddings instead of
    plain CLS tokens. Requires --sam_checkpoint.

build options:
    --scene SCENE_ID         Scene ID (required)
    --model MODEL            Embedding model (default: dinov2)
    --data_root PATH         Root directory for robocasa data
    --rebuild                Force rebuild even if index already exists
    --batch_size N           Batch size (default: 32)
    --device DEVICE          cuda or cpu (default: auto)
    --with_masks             Enable Stage B Mask Inversion (recommended)
    --sam_checkpoint PATH    Path to SAM .pth checkpoint (required with --with_masks)
    --regen_masks            Regenerate masks even if captions.pt already exists

query options:
    --scene SCENE_ID         Scene ID (required)
    --query PATH             Query image path(s); repeat for multi-view averaging
    --model MODEL            Must match the model used for build (default: dinov2)
    --data_root PATH         Root directory for robocasa data
    --top_k K                Number of top results to return (default: 10)
    --output PATH            Where to save the result grid image
    --show                   Open image viewer after saving
    --device DEVICE          cuda or cpu
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── SoIR path ─────────────────────────────────────────────────────────────────
SOIR_DIR = os.path.expanduser("~/Projects/SoIR")
if SOIR_DIR not in sys.path:
    sys.path.insert(0, SOIR_DIR)

SAM_CHECKPOINT_DEFAULT = os.path.expanduser(
    "~/Projects/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
)

# ── model registry ────────────────────────────────────────────────────────────
# name → (plain_module, plain_class, mi_module, mi_class, vec_dim, B_model)
MODELS = {
    "dinov2":      ("extractors.dinov2_extractor",    "DinoV2Extractor",
                    "extractors.dinov2_mi_extractor",  "DinoV2MIExtractor",  1024, False),
    "dinov2_base": ("extractors.dinov2_extractor",    "DinoV2Extractor",
                    "extractors.dinov2_mi_extractor",  "DinoV2MIExtractor",   768, True),
    "clip":        ("extractors.clip_extractor",      "CLIPExtractor",
                    "extractors.clip_mi_extractor",    "CLIPMIExtractor",     512, False),
    "siglip":      ("extractors.siglip_extractor",    "SigLIPExtractor",
                    "extractors.siglip_mi_extractor",  "SigLIPMIExtractor",  1152, False),
    "siglip_base": ("extractors.siglip_extractor",    "SigLIPExtractor",
                    "extractors.siglip_mi_extractor",  "SigLIPMIExtractor",   768, True),
}
MODEL_CHOICES = list(MODELS.keys())

# ── index paths ───────────────────────────────────────────────────────────────
INDEX_BIN         = "index.bin"
PATHS_JSON        = "frame_paths.json"
MASK_INDICES_JSON = "mask_indices.json"
CAPTIONS_PT       = "captions.pt"


def get_index_dir(data_root: str, scene_id: str, model: str, with_masks: bool) -> Path:
    suffix = "_mao" if with_masks else ""
    return Path(data_root) / scene_id / f"retrieval_index_{model}{suffix}"


def get_captions_path(data_root: str, scene_id: str) -> Path:
    return Path(data_root) / scene_id / CAPTIONS_PT


def get_frame_paths(data_root: str, scene_id: str) -> list[str]:
    color_dir = Path(data_root) / scene_id / "color"
    if not color_dir.is_dir():
        raise FileNotFoundError(f"Color frames directory not found: {color_dir}")
    paths = sorted(str(p) for p in color_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {color_dir}")
    return paths


# ── mask generation (OWLv2 + SAM → captions.pt) ──────────────────────────────

def generate_masks(frame_paths: list[str], captions_path: Path,
                   sam_checkpoint: str, device: str) -> None:
    """Run OWLv2 objectness detection + SAM segmentation on all scene frames."""
    print("Loading OWLv2 detector …")
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from torchvision import ops

    owlv2_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlv2_model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble"
    ).to(device).eval()

    print(f"Loading SAM (vit_h) from {sam_checkpoint} …")
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    def detect_objects(image: Image.Image, threshold: float = 0.2):
        """OWLv2 objectness-only detection."""
        px = owlv2_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            fmap, _ = owlv2_model.image_embedder(
                pixel_values=px, output_attentions=False, output_hidden_states=False
            )
            s = fmap.shape
            fmap_flat = fmap.reshape(s[0], s[1] * s[2], s[3])
            boxes = owlv2_model.box_predictor(fmap_flat, fmap)
            scores = torch.sigmoid(owlv2_model.objectness_predictor(image_features=fmap_flat)[0])
        # convert boxes: center→corners, relative→absolute
        cx, cy, w, h = boxes[0].unbind(-1)
        side = max(image.size)
        x1 = (cx - w / 2) * side; y1 = (cy - h / 2) * side
        x2 = (cx + w / 2) * side; y2 = (cy + h / 2) * side
        abs_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        keep = scores > threshold
        abs_boxes, scores = abs_boxes[keep].cpu(), scores[keep].cpu()
        if len(abs_boxes) > 0:
            nms_idx = ops.nms(abs_boxes.float(), scores, iou_threshold=0.4)
            abs_boxes = abs_boxes[nms_idx]
        return abs_boxes

    captions = {}
    n = len(frame_paths)
    for i, path in enumerate(frame_paths):
        print(f"  Generating masks {i+1}/{n}: {os.path.basename(path)}", end="\r", flush=True)
        img = Image.open(path).convert("RGB")
        bboxes = detect_objects(img)

        masks = []
        if len(bboxes) > 0:
            predictor.set_image(np.array(img))
            for bbox in bboxes:
                x0, y0, x1, y1 = bbox.int().tolist()
                mask_result, _, _ = predictor.predict(
                    box=np.array([[x0, y0, x1, y1]]), multimask_output=False
                )
                # Convert bool mask → uint8 0/255, store at 1/4 resolution.
                # Extractors resize internally (to 224 for DINOv2, 384 for SigLIP).
                m = mask_result[0].astype(np.uint8) * 255
                h4, w4 = max(1, m.shape[0] // 4), max(1, m.shape[1] // 4)
                masks.append(np.array(Image.fromarray(m).resize((w4, h4), Image.NEAREST)))

        if len(masks) == 0:
            h4, w4 = max(1, img.height // 4), max(1, img.width // 4)
            masks = [np.full((h4, w4), 255, dtype=np.uint8)]

        # Use 'masks' key — this is the format both DinoV2Extractor and
        # SigLIPExtractor.extract_mask_from_path() expect.
        captions[path] = {"masks": masks}

    print()
    torch.save(captions, str(captions_path))
    avg = sum(len(v['masks']) for v in captions.values()) // n
    print(f"  Masks saved to {captions_path}  ({n} frames, avg {avg} objects/frame)")


# ── extractor factory ─────────────────────────────────────────────────────────

def make_extractor(model_name: str, device: str, captions_file: str = None):
    import importlib
    from munch import Munch

    plain_mod, plain_cls, mi_mod, mi_cls, vec_dim, b_model = MODELS[model_name]

    use_mi = captions_file is not None
    mod_path = mi_mod if use_mi else plain_mod
    cls_name = mi_cls if use_mi else plain_cls

    module = importlib.import_module(mod_path)
    cls = getattr(module, cls_name)

    args = Munch(
        B_model=b_model,
        lora_adapt=False,
        weights=None,
        captions_file=captions_file,
        wo_norm_features=False,
        mi_alpha=0.03,
        mi_iterations=100,
        mi_layer_index=-1,
        global_features=True,
        smart_crop=False,
        mi_sum=False,
        mask_input=False,
        full_mask=False,
    )
    extractor = cls(args=args, device=device)

    mode = "MaO (Stage B)" if use_mi else "plain CLS"
    print(f"  Model: {model_name}  ({vec_dim}-d)  mode={mode}  device={device}")
    return extractor


def extract_features(extractor, image_paths: list[str],
                     batch_size: int = 32,
                     is_query: bool = False,
                     max_masks_per_frame: int = 5) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Extract L2-normalised embeddings following create_index.py.

    Returns (embeddings [M, D], embed_paths [M], embed_mask_indices [M]):
    a frame with N objects contributes up to max_masks_per_frame vectors;
    embed_mask_indices[i] is the mask index within that frame for FAISS entry i.

    Masks are processed one at a time to avoid GPU OOM.
    """
    all_feats = []
    all_paths = []
    all_mask_indices = []

    # Check if extractor has loaded captions (MI mode)
    has_captions = (hasattr(extractor, 'captions') and
                    extractor.captions is not None)

    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start:start + batch_size]
        for p in chunk:
            img = Image.open(p).convert("RGB")

            if has_captions and p in extractor.captions:
                frame_masks = extractor.captions[p].get('masks', [])
            else:
                frame_masks = []

            # Plain CLS: single call, no masks
            if not frame_masks:
                extractor.is_current_query = is_query
                out = extractor.extract({"image": img, "img_path": p}, is_query=is_query)
                feat = out["keypoints"]
                if feat.ndim == 1:
                    feat = feat.unsqueeze(0)
                feat = F.normalize(feat[:1], p=2, dim=-1)
                all_feats.append(feat.cpu().float())
                all_paths.append(p)
                all_mask_indices.append(0)
                continue

            # MI mode: one mask at a time to stay within GPU memory
            for midx, mask in enumerate(frame_masks[:max_masks_per_frame]):
                extractor.is_current_query = is_query
                # Pass img_path=None so the extractor won't reload all masks;
                # supply the single mask directly via the masks= kwarg.
                out = extractor.extract(
                    {"image": img, "img_path": None},
                    is_query=is_query,
                    masks=[mask],
                )
                feat = out["keypoints"]
                if feat.ndim == 1:
                    feat = feat.unsqueeze(0)
                feat = F.normalize(feat[:1], p=2, dim=-1)
                all_feats.append(feat.cpu().float())
                all_paths.append(p)
                all_mask_indices.append(midx)

        done = min(start + batch_size, len(image_paths))
        print(f"  {done}/{len(image_paths)} frames processed", end="\r", flush=True)
    print()
    return torch.cat(all_feats, dim=0).detach().numpy(), all_paths, all_mask_indices


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray):
    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def load_faiss_index(path: str):
    import faiss
    return faiss.read_index(path)

def save_faiss_index(index, path: str):
    import faiss
    faiss.write_index(index, path)


# ── visualisation ─────────────────────────────────────────────────────────────

def make_grid(images: list[np.ndarray], labels: list[str],
              n_cols: int = 5, thumb_size: int = 240) -> np.ndarray:
    from PIL import ImageDraw, ImageFont
    n_rows = math.ceil(len(images) / n_cols)
    cell_h = thumb_size + 24
    grid = Image.new("RGB", (n_cols * thumb_size, n_rows * cell_h), (240, 240, 240))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    for i, (img_arr, label) in enumerate(zip(images, labels)):
        row, col = divmod(i, n_cols)
        x, y = col * thumb_size, row * cell_h
        thumb = Image.fromarray(img_arr).resize((thumb_size, thumb_size), Image.LANCZOS)
        grid.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb_size + 4), label, fill=(30, 30, 30), font=font)
    return np.array(grid)


# ── build ─────────────────────────────────────────────────────────────────────

def cmd_build(args):
    index_dir  = get_index_dir(args.data_root, args.scene, args.model, args.with_masks)
    index_path = str(index_dir / INDEX_BIN)
    paths_path = str(index_dir / PATHS_JSON)

    if not args.rebuild and os.path.exists(index_path) and os.path.exists(paths_path):
        if not args.incremental:
            print(f"Index already exists at {index_dir}. Use --rebuild to overwrite.")
            return

    print(f"Collecting frames for scene '{args.scene}' …")
    frame_paths = get_frame_paths(args.data_root, args.scene)
    print(f"  Found {len(frame_paths)} frames.")

    # ── Incremental mode: only process frames not yet in the index ────────────
    if args.incremental and not args.rebuild and \
            os.path.exists(index_path) and os.path.exists(paths_path):
        with open(paths_path) as _f:
            _existing = json.load(_f)
        _existing_set = set(_existing)
        _new = [p for p in frame_paths if p not in _existing_set]
        if not _new:
            print(f"  Incremental: no new frames to add ({len(_existing_set)} already indexed).")
            return
        print(f"  Incremental: {len(_new)} new frames  (+{len(_new)} / {len(_existing_set)} existing)")
        extractor = make_extractor(args.model, args.device, captions_file=None)
        new_emb, new_paths, new_midx = extract_features(
            extractor, _new, batch_size=args.batch_size, is_query=False,
            max_masks_per_frame=args.max_masks_per_frame,
        )
        # append to existing FAISS index (IndexFlatIP supports add())
        _old_idx = load_faiss_index(index_path)
        _old_idx.add(new_emb)
        save_faiss_index(_old_idx, index_path)
        # update frame_paths.json
        _updated = _existing + new_paths
        with open(paths_path, "w") as _f:
            json.dump(_updated, _f)
        # update mask_indices.json
        _midx_path = str(index_dir / MASK_INDICES_JSON)
        if os.path.exists(_midx_path):
            with open(_midx_path) as _f:
                _old_midx = json.load(_f)
        else:
            _old_midx = [0] * len(_existing)
        with open(_midx_path, "w") as _f:
            json.dump(_old_midx + new_midx, _f)
        print(f"  Incremental index updated: {len(_existing_set)} → {len(_updated)} paths  "
              f"({_old_idx.ntotal} vectors)")
        return

    # ── mask generation ───────────────────────────────────────────────────────
    captions_file = None
    if args.with_masks:
        captions_path = get_captions_path(args.data_root, args.scene)
        if args.regen_masks or not captions_path.exists():
            if not args.sam_checkpoint or not os.path.exists(args.sam_checkpoint):
                print(f"ERROR: --sam_checkpoint not found: {args.sam_checkpoint}")
                sys.exit(1)
            print("Generating object masks (OWLv2 + SAM) …")
            generate_masks(frame_paths, captions_path, args.sam_checkpoint, args.device)
        else:
            print(f"Using existing masks: {captions_path}")
        captions_file = str(captions_path)

    # ── feature extraction ────────────────────────────────────────────────────
    print("Loading extractor …")
    extractor = make_extractor(args.model, args.device, captions_file=captions_file)

    print("Extracting features …")
    embeddings, embed_paths, embed_mask_indices = extract_features(
        extractor, frame_paths, batch_size=args.batch_size, is_query=False,
        max_masks_per_frame=args.max_masks_per_frame,
    )
    print(f"  {embeddings.shape[0]} vectors for {len(frame_paths)} frames")

    index_dir.mkdir(parents=True, exist_ok=True)
    save_faiss_index(build_faiss_index(embeddings), index_path)
    with open(paths_path, "w") as f:
        json.dump(embed_paths, f)
    with open(str(index_dir / MASK_INDICES_JSON), "w") as f:
        json.dump(embed_mask_indices, f)
    print(f"Index saved to {index_dir}")


# ── Gemini reranking ──────────────────────────────────────────────────────────

def rerank_with_gemini(query_paths: list[str], candidate_paths: list[str],
                       candidate_scores: list[float],
                       model_name: str = "gemini-2.5-flash",
                       api_key: str = None) -> list[int]:
    """
    Ask Gemini to rerank candidate frames by how clearly the query object appears.

    Returns a list of 0-based indices into candidate_paths in reranked order.
    Falls back to original order on any error.
    """
    import google.generativeai as genai

    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        print("WARNING: no Gemini API key found (set GOOGLE_API_KEY or pass --gemini_api_key). Skipping rerank.")
        return list(range(len(candidate_paths)))

    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)

    def _pil(path):
        return Image.open(path).convert("RGB")

    # Build the prompt content: query image(s) then numbered candidates
    parts = []
    if len(query_paths) == 1:
        parts.append("This is the query image showing an object I am looking for:\n")
    else:
        parts.append(f"These {len(query_paths)} images show the object I am looking for from different angles:\n")
    for p in query_paths:
        parts.append(_pil(p))

    parts.append(
        f"\nBelow are {len(candidate_paths)} candidate scene frames, numbered 1 to {len(candidate_paths)}. "
        "Each image is a frame from a scene that may or may not contain the same object.\n"
        "Please rerank the frames from most likely to contain the query object (clearly visible) "
        "to least likely. Consider visibility, size, and similarity to the query object.\n"
        "Respond with ONLY a comma-separated list of the frame numbers in your preferred order, "
        "e.g.: 3,1,5,2,4\n"
    )
    for i, p in enumerate(candidate_paths):
        parts.append(f"\nFrame {i+1} (FAISS score {candidate_scores[i]:.4f}):\n")
        parts.append(_pil(p))

    print(f"Calling Gemini ({model_name}) to rerank {len(candidate_paths)} candidates …")
    try:
        response = model.generate_content(parts)
        raw = response.text.strip()
        print(f"  Gemini response: {raw}")
        # Parse comma-separated 1-based indices → 0-based
        reranked = []
        seen = set()
        for tok in raw.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            # strip any non-digit prefix/suffix (e.g. "Frame 3" → "3")
            digits = "".join(c for c in tok if c.isdigit())
            if not digits:
                continue
            idx = int(digits) - 1
            if 0 <= idx < len(candidate_paths) and idx not in seen:
                reranked.append(idx)
                seen.add(idx)
        # Append any missing indices at the end (Gemini may have omitted some)
        for i in range(len(candidate_paths)):
            if i not in seen:
                reranked.append(i)
        return reranked
    except Exception as e:
        print(f"WARNING: Gemini rerank failed ({e}). Using original order.")
        return list(range(len(candidate_paths)))


# ── text embedding helper ─────────────────────────────────────────────────────

def _embed_text(extractor, text: str) -> "np.ndarray":
    """Encode a text string into a unit-norm embedding using the extractor's
    text encoder.  Works for CLIPExtractor and SigLIPExtractor.

    Returns a (1, D) float32 numpy array ready for FAISS IndexFlatIP search.
    Raises ValueError for vision-only models (DINOv2).
    """
    if not hasattr(extractor, "_encode_text"):
        raise ValueError(
            "The loaded extractor does not support text queries. "
            "Use --model clip, siglip, or siglip_base."
        )
    raw = extractor._encode_text([text])

    # CLIPExtractor returns a plain tensor (1, D).
    # SigLIPExtractor returns BaseModelOutputWithPooling.
    if isinstance(raw, torch.Tensor):
        feat = raw
    elif hasattr(raw, "pooler_output") and raw.pooler_output is not None:
        feat = raw.pooler_output          # SigLIP: (1, D)
    elif hasattr(raw, "last_hidden_state"):
        feat = raw.last_hidden_state[:, 0, :]   # CLS token fallback
    else:
        raise ValueError(f"Unexpected _encode_text output type: {type(raw)}")

    return F.normalize(feat, p=2, dim=-1).cpu().float().numpy()   # (1, D)


# ── HyDE-style text query helpers ────────────────────────────────────────────

def rewrite_text_queries(query_text: str, n_queries: int = 4,
                         api_key: str | None = None,
                         model_name: str = "gemini-2.5-flash-lite") -> list[str]:
    """Generate n diverse visual re-phrasings of query_text using Gemini.

    Each re-phrasing describes the same object from a different angle, context,
    or level of detail so that together they cover more of the embedding space
    (HyDE-style: diverse hypothetical descriptions instead of documents).
    Falls back to [query_text] on any error.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        print("  [HyDE] No Gemini key — using original text only.")
        return [query_text]
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model_name)
        prompt = (
            f'I want to find an object in robot indoor scene images using text-image retrieval.\n'
            f'The user is looking for: "{query_text}"\n\n'
            f'Interpret the user\'s intent and generate {n_queries} diverse short search phrases '
            f'(2-5 words each, noun phrases only, no full sentences) that describe what the '
            f'object looks like visually. '
            f'Vary material, colour, style, or context. '
            f'Example: if the user says "something to wear", write phrases like '
            f'"jacket", "folded clothes", "shirt on surface".\n'
            f'Return ONLY the {n_queries} phrases, one per line, no numbers or bullets.'
        )
        resp = m.generate_content(prompt)
        lines = [ln.strip() for ln in resp.text.strip().splitlines() if ln.strip()]
        queries = lines[:n_queries]
        # Pad with the original text if Gemini returned fewer lines than requested
        while len(queries) < n_queries:
            queries.append(query_text)
        print(f"  [HyDE] Query rewrites:")
        for i, q in enumerate(queries):
            print(f"    [{i+1}] {q}")
        return queries
    except Exception as e:
        print(f"  [HyDE] Rewrite failed ({e}) — using original text.")
        return [query_text]


def rerank_with_gemini_text(query_text: str,
                             candidate_paths: list[str],
                             candidate_scores: list[float],
                             api_key: str | None = None,
                             model_name: str = "gemini-2.5-flash") -> list[int]:
    """Rerank candidate frames by showing them to Gemini together with the text
    description (no reference image needed).

    Returns 0-based indices in reranked order; falls back to original order.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        return list(range(len(candidate_paths)))
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model_name)

        n_c = len(candidate_paths)
        parts: list = [
            f'I am looking for this object: "{query_text}"\n\n'
            f'Below are {n_c} candidate scene frames numbered 1 to {n_c}. '
            f'Rerank them from most likely to contain the target object (clearly '
            f'visible and reachable) to least likely.\n'
            f'Respond with ONLY a comma-separated list of frame numbers, e.g.: 3,1,5,2,4\n'
        ]
        for i, path in enumerate(candidate_paths):
            parts.append(f'\nFrame {i+1} (score={candidate_scores[i]:.4f}):\n')
            parts.append(Image.open(path).convert("RGB"))

        print(f"Calling Gemini ({model_name}) to rerank {n_c} candidates by text …")
        resp = m.generate_content(parts)
        raw  = resp.text.strip()
        print(f"  Gemini text-rerank response: {raw}")

        reranked, seen = [], set()
        for tok in raw.replace(";", ",").split(","):
            digits = "".join(c for c in tok.strip() if c.isdigit())
            if not digits:
                continue
            idx = int(digits) - 1
            if 0 <= idx < n_c and idx not in seen:
                reranked.append(idx); seen.add(idx)
        for i in range(n_c):
            if i not in seen:
                reranked.append(i)
        return reranked
    except Exception as e:
        print(f"WARNING: Gemini text rerank failed ({e}). Using original order.")
        return list(range(len(candidate_paths)))


def verify_with_gemini_text(query_text: str,
                             candidate_frames: list[dict],
                             api_key: str | None = None,
                             model_name: str = "gemini-2.5-flash") -> int:
    """Final VLM verification: from the top-k reranked candidates pick the
    single frame that best shows the target object in a reachable position.

    candidate_frames: list of dicts with keys frame_path, score, x, y.
    Returns 0-based index into candidate_frames (0 on any error).
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        return 0
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model_name)

        n_c = len(candidate_frames)
        parts: list = [
            f'I am searching for this object with a robot: "{query_text}"\n\n'
            f'Below are {n_c} candidate frames (numbered 1–{n_c}) from a robot scan.\n'
            f'Which frame number most clearly shows the target object in a position '
            f'the robot can reach?\n'
            f'Answer with ONLY the frame number (1–{n_c}), or 0 if none show it.\n'
        ]
        for i, frame in enumerate(candidate_frames):
            parts.append(f'\nFrame {i+1} (score={frame["score"]:.4f}):\n')
            parts.append(Image.open(frame["frame_path"]).convert("RGB"))

        print(f"Calling Gemini ({model_name}) for final text verification on {n_c} frames …")
        resp = m.generate_content(parts)
        raw  = resp.text.strip()
        print(f"  Gemini text-verify response: {raw}")

        digits = "".join(c for c in raw if c.isdigit())
        if digits:
            idx = int(digits) - 1
            if 0 <= idx < n_c:
                print(f"  → chose frame {idx+1}: "
                      f"{os.path.basename(candidate_frames[idx]['frame_path'])}")
                return idx
        print("  → could not parse; using frame 1")
        return 0
    except Exception as e:
        print(f"WARNING: Gemini text verification failed ({e}). Using top-1.")
        return 0


# ── query ─────────────────────────────────────────────────────────────────────

def cmd_query(args):
    if not args.query and not getattr(args, "query_text", None):
        print("ERROR: provide --query IMAGE(s) or --query_text TEXT")
        sys.exit(1)

    # Use explicitly requested index type, or auto-detect
    if args.with_masks is not None:
        index_dir = get_index_dir(args.data_root, args.scene, args.model, args.with_masks)
    else:
        # Try MaO index first, fall back to plain
        index_dir = get_index_dir(args.data_root, args.scene, args.model, with_masks=True)
        if not (index_dir / INDEX_BIN).exists():
            index_dir = get_index_dir(args.data_root, args.scene, args.model, with_masks=False)
    index_path = str(index_dir / INDEX_BIN)
    paths_path = str(index_dir / PATHS_JSON)

    if not os.path.exists(index_path):
        print(f"ERROR: index not found. Run: retrieve build --scene {args.scene}")
        sys.exit(1)

    print(f"Loading FAISS index from {index_dir} …")
    faiss_index = load_faiss_index(index_path)
    with open(paths_path) as f:
        embed_paths = json.load(f)
    mask_indices_path = str(index_dir / MASK_INDICES_JSON)
    if os.path.exists(mask_indices_path):
        with open(mask_indices_path) as f:
            embed_mask_indices = json.load(f)
    else:
        embed_mask_indices = [0] * len(embed_paths)
    n_unique = len(set(embed_paths))
    print(f"  {faiss_index.ntotal} vectors  ({n_unique} unique frames)")

    # Load captions.pt for mask overlay — only meaningful with MaO index
    captions = None
    using_mao = "_mao" in str(index_dir)
    if using_mao:
        captions_path = get_captions_path(args.data_root, args.scene)
        if captions_path.exists():
            print(f"  Loading masks from {captions_path} for overlay …")
            captions = torch.load(str(captions_path), weights_only=False)

    print("Loading extractor for query …")
    # Query always uses plain CLS (no masks needed for rendered object images)
    extractor = make_extractor(args.model, args.device, captions_file=None)

    query_paths  = args.query or []
    query_text   = getattr(args, "query_text", None)
    using_text   = bool(query_text) and not query_paths
    rewritten: list[str] = []   # populated below for text queries

    if using_text:
        print(f"Text query: {query_text!r}  (model={args.model})")
        if args.model.startswith("dinov2"):
            print("ERROR: DINOv2 is vision-only and does not support text queries. "
                  "Use --model clip, siglip, or siglip_base.")
            sys.exit(1)
    else:
        print(f"Extracting query features from {len(query_paths)} image(s) …")

    k_search = min(args.top_k, faiss_index.ntotal)
    pooled: dict[str, tuple[float, int]] = {}  # path → (best_score, mask_idx)

    if using_text:
        # ── HyDE text pipeline ────────────────────────────────────────────────
        # 1. Rewrite to N diverse visual descriptions
        rewritten = rewrite_text_queries(
            query_text, n_queries=3,
            api_key=getattr(args, "gemini_api_key", None),
        )  # model_name left at default "gemini-2.5-flash-lite" (fast, cheap)
        rewritten.append(query_text)   # always include the original as the 4th query
        # 2. Embed each rewrite, take top_k each, union (dedup by best score)
        k_per = min(args.top_k, faiss_index.ntotal)
        for qi, rq in enumerate(rewritten):
            try:
                q_feat = _embed_text(extractor, rq)
                scores_i, indices_i = faiss_index.search(q_feat, k_per)
                for idx, score in zip(indices_i[0], scores_i[0]):
                    path     = embed_paths[int(idx)]
                    mask_idx = embed_mask_indices[int(idx)]
                    prev     = pooled.get(path, (float("-inf"), 0))[0]
                    if float(score) > prev:
                        pooled[path] = (float(score), mask_idx)
                print(f"  rewrite {qi+1}/{len(rewritten)} ({rq!r}): "
                      f"{len(pooled)} unique candidates pooled")
            except Exception as _e:
                print(f"  rewrite {qi+1}/{len(rewritten)} ({rq!r}): FAILED — {_e}")
    else:
        # ── Image query: one FAISS search per query image ─────────────────────
        for qi, p in enumerate(query_paths):
            img = Image.open(p).convert("RGB")
            out = extractor.extract(img, is_query=True)
            feat = out["keypoints"]
            if feat.ndim == 1:
                feat = feat.unsqueeze(0)
            q_feat = F.normalize(
                feat.mean(0, keepdim=True), p=2, dim=-1
            ).cpu().float().numpy()

            scores_i, indices_i = faiss_index.search(q_feat, k_search)
            for idx, score in zip(indices_i[0], scores_i[0]):
                path     = embed_paths[int(idx)]
                mask_idx = embed_mask_indices[int(idx)]
                prev     = pooled.get(path, (float("-inf"), 0))[0]
                if float(score) > prev:
                    pooled[path] = (float(score), mask_idx)
            print(f"  query {qi+1}/{len(query_paths)}  ({os.path.basename(p)}): "
                  f"{len(pooled)} unique candidates so far")

    # For text: pool already contains at most top_k * n_rewrites entries; take all.
    # For image: keep top_k per query image.
    if using_text:
        sorted_paths = sorted(pooled, key=lambda p: pooled[p][0], reverse=True)
    else:
        n_keep = args.top_k * max(len(query_paths), 1)
        sorted_paths = sorted(pooled, key=lambda p: pooled[p][0], reverse=True)[:n_keep]

    result_paths        = sorted_paths
    result_scores       = [pooled[p][0] for p in result_paths]
    result_mask_indices = [pooled[p][1] for p in result_paths]

    n_sources = len(rewritten) if using_text else len(query_paths)
    print(f"\nPooled {len(result_paths)} candidates from {n_sources} queries  [{args.model}]:")
    for rank, (path, score, midx) in enumerate(zip(result_paths, result_scores, result_mask_indices)):
        print(f"  #{rank+1:2d}  score={score:.4f}  mask={midx}  {os.path.basename(path)}")

    # ── Gemini reranking (runs before pose output so top-1 reflects final order) ─
    if using_text and result_paths and not getattr(args, "skip_rerank", False):
        # Text pipeline: rerank + verify (skipped when --skip_rerank is set).
        gemini_key   = getattr(args, "gemini_api_key", None)
        gemini_model = getattr(args, "gemini_model", "gemini-2.5-flash")

        # Step A: rerank all candidates by text description
        reranked_order = rerank_with_gemini_text(
            query_text, result_paths, result_scores,
            api_key=gemini_key, model_name=gemini_model,
        )
        result_paths  = [result_paths[i]  for i in reranked_order]
        result_scores = [result_scores[i] for i in reranked_order]
        result_mask_indices = [result_mask_indices[i] for i in reranked_order]
        print(f"\nReranked order [Gemini text]:")
        for rank, (path, score) in enumerate(zip(result_paths, result_scores)):
            print(f"  #{rank+1:2d}  score={score:.4f}  {os.path.basename(path)}")

        # Step B: VLM verification — pick single best from top-5 reranked frames
        verify_k = min(5, len(result_paths))
        verify_frames = []
        for path, score in zip(result_paths[:verify_k], result_scores[:verify_k]):
            verify_frames.append({"frame_path": path, "score": float(score)})
        best_idx = verify_with_gemini_text(
            query_text, verify_frames,
            api_key=gemini_key, model_name="gemini-2.5-pro",
        )
        # Promote verified best to position 0 without dropping the rest
        if best_idx > 0:
            result_paths  = [result_paths[best_idx]]  + result_paths[:best_idx]  + result_paths[best_idx+1:]
            result_scores = [result_scores[best_idx]] + result_scores[:best_idx] + result_scores[best_idx+1:]
            result_mask_indices = ([result_mask_indices[best_idx]]
                                   + result_mask_indices[:best_idx]
                                   + result_mask_indices[best_idx+1:])
        print(f"  Final top-1 after verification: {os.path.basename(result_paths[0])}")
    elif using_text:
        print("WARNING: text query returned no results. "
              "Check that a siglip/clip index exists for this scene.")

    elif args.rerank:
        print(f"\nSending all {len(result_paths)} candidates + {len(query_paths)} query views to Gemini …")
        reranked_order = rerank_with_gemini(
            query_paths, result_paths, result_scores,
            model_name=args.gemini_model,
            api_key=args.gemini_api_key,
        )
        result_paths  = [result_paths[i]  for i in reranked_order]
        result_scores = [result_scores[i] for i in reranked_order]
        result_mask_indices = [result_mask_indices[i] for i in reranked_order]
        print(f"\nReranked order  [Gemini]:")
        for rank, (path, score) in enumerate(zip(result_paths, result_scores)):
            print(f"  #{rank+1:2d}  faiss={score:.4f}  {os.path.basename(path)}")

    # ── Output top-1 pose + top-k candidates for navigation integration ─────────
    # Runs after reranking so result_paths[0] is always the final #1 pick.
    if not result_paths:
        print("ERROR: no results found — nothing to navigate to.")
        if getattr(args, "output_pose", None):
            import json as _json
            with open(args.output_pose, "w") as f:
                _json.dump({"frame_path": None, "score": 0.0,
                            "x": None, "y": None, "yaw": None,
                            "top_k_frames": []}, f, indent=2)
        # Still save the grid so the user can see the placeholder
    if result_paths and getattr(args, "output_pose", None):
        import json as _json

        def _load_pose(frame_path: str) -> tuple:
            """Return (x, y, yaw) for a scan frame, or (None, None, None)."""
            scene_dir = Path(frame_path).parent.parent
            stem = Path(frame_path).stem
            xy_file = scene_dir / "robot_xy" / f"{stem}.txt"
            if xy_file.exists():
                xyw = np.loadtxt(str(xy_file)).flatten()
                x, y = float(xyw[0]), float(xyw[1])
                yaw = float(xyw[2]) if len(xyw) >= 3 else None
                return x, y, yaw
            pose_file = scene_dir / "pose" / f"{stem}.txt"
            if pose_file.exists():
                cam2world = np.loadtxt(str(pose_file))
                tx, ty = float(cam2world[0, 3]), float(cam2world[1, 3])
                if abs(tx) > 1e-3 or abs(ty) > 1e-3:
                    return tx, ty, None
            return None, None, None

        top_x, top_y, top_yaw = _load_pose(result_paths[0])
        if top_x is None:
            print(f"  WARNING: no robot_xy/ or pose/ file found for frame {Path(result_paths[0]).stem}")

        # Collect poses for top-k candidates (capped at args.top_k so the
        # verification VLM call only sees the intended number of images).
        top_k_frames = []
        for path, score in zip(result_paths[:args.top_k], result_scores[:args.top_k]):
            fx, fy, fyaw = _load_pose(path)
            top_k_frames.append({
                "frame_path": path,
                "score":      float(score),
                "x":          fx,
                "y":          fy,
                "yaw":        fyaw,
            })

        pose_data = {
            "frame_path":   result_paths[0],
            "score":        float(result_scores[0]),
            "x":            top_x,
            "y":            top_y,
            "yaw":          top_yaw,
            "top_k_frames": top_k_frames,
        }
        with open(args.output_pose, "w") as f:
            _json.dump(pose_data, f, indent=2)
        xy_str = f"  xy=({top_x:.3f}, {top_y:.3f})" if top_x is not None else "  xy=UNKNOWN (rescan needed)"
        print(f"  Top-1 pose saved → {args.output_pose}{xy_str}  ({len(top_k_frames)} frames in top_k_frames)")

    print("\nBuilding result grid …")
    if using_text:
        # For text queries there is no reference image; make a placeholder tile.
        from PIL import ImageDraw as _ID, ImageFont as _IF
        _tile = Image.new("RGB", (240, 240), (230, 240, 255))
        _draw = _ID.Draw(_tile)
        try:
            _font = _IF.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            _font = _IF.load_default()
        # Wrap text manually for narrow tile
        words = query_text.split()
        lines, line = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 18:
                lines.append(" ".join(line[:-1])); line = [w]
        if line: lines.append(" ".join(line))
        y0 = max(0, (240 - len(lines) * 18) // 2)
        for i, ln in enumerate(lines):
            _draw.text((8, y0 + i * 18), ln, fill=(30, 30, 120), font=_font)
        q_images = [np.array(_tile)]
        q_labels = [f'TEXT: "{query_text[:20]}"']
    else:
        q_images = [np.array(Image.open(p).convert("RGB")) for p in query_paths]
        q_labels = [f"QUERY {i+1}" for i in range(len(q_images))]

    r_images = []
    r_labels = []
    crop_images = []
    crop_labels = []
    for rank, (path, score, midx) in enumerate(zip(result_paths, result_scores, result_mask_indices)):
        frame = np.array(Image.open(path).convert("RGB"))
        # Overlay the matched object mask in red if captions are available
        crop_img = None
        if captions is not None and path in captions:
            masks_list = captions[path].get("masks", [])
            if midx < len(masks_list):
                mask_small = masks_list[midx]
                ms = np.array(mask_small).astype(np.uint8)
                mask_full = np.array(
                    Image.fromarray(ms).resize(
                        (frame.shape[1], frame.shape[0]), Image.NEAREST
                    )
                )
                obj_pixels = mask_full > 0
                frame = frame.copy()
                # Semi-transparent red fill
                frame[obj_pixels] = (
                    frame[obj_pixels] * 0.45 + np.array([255, 60, 60]) * 0.55
                ).astype(np.uint8)
                # Thick red bounding box (visible even at thumbnail scale)
                rows_hit = np.where(obj_pixels.any(axis=1))[0]
                cols_hit = np.where(obj_pixels.any(axis=0))[0]
                if len(rows_hit) > 0 and len(cols_hit) > 0:
                    r0, r1 = int(rows_hit[0]), int(rows_hit[-1])
                    c0, c1 = int(cols_hit[0]), int(cols_hit[-1])
                    t = max(4, frame.shape[0] // 60)  # ~1.7% of frame height
                    H, W = frame.shape[:2]
                    frame[max(0,r0-t):r0+t, max(0,c0-t):min(W,c1+t)] = [255, 0, 0]
                    frame[r1-t:min(H,r1+t), max(0,c0-t):min(W,c1+t)] = [255, 0, 0]
                    frame[max(0,r0-t):min(H,r1+t), max(0,c0-t):c0+t] = [255, 0, 0]
                    frame[max(0,r0-t):min(H,r1+t), c1-t:min(W,c1+t)] = [255, 0, 0]
                    # Zoomed crop of the matched region (with padding)
                    pad = max(20, (r1 - r0) // 2, (c1 - c0) // 2)
                    cr0, cr1 = max(0, r0-pad), min(H, r1+pad)
                    cc0, cc1 = max(0, c0-pad), min(W, c1+pad)
                    crop_img = frame[cr0:cr1, cc0:cc1]
        r_images.append(frame)
        r_labels.append(f"#{rank+1} s={score:.3f} m={midx}")
        crop_images.append(crop_img if crop_img is not None
                           else np.full((10, 10, 3), 200, dtype=np.uint8))
        crop_labels.append(f"#{rank+1} crop")

    sep = np.full((240, 240, 3), 200, dtype=np.uint8)
    all_images = q_images + [sep] + r_images + [sep] + crop_images
    all_labels = q_labels + ["---"] + r_labels + ["---"] + crop_labels
    grid = make_grid(all_images, all_labels, n_cols=5)


    import imageio
    imageio.imwrite(args.output, grid)
    print(f"Grid saved to {args.output}")

    if args.show:
        try:
            import subprocess
            subprocess.Popen(["xdg-open", args.output])
        except Exception:
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve robocasa scene frames matching a query object image"
    )
    sub = parser.add_subparsers(dest="subcommand")

    def add_common(p):
        p.add_argument("--scene",     required=True)
        p.add_argument("--model",     default="dinov2", choices=MODEL_CHOICES)
        p.add_argument("--data_root", default=os.path.expanduser("~/Projects/robocasa_data"))
        p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")

    p_build = sub.add_parser("build")
    add_common(p_build)
    p_build.add_argument("--rebuild",         action="store_true")
    p_build.add_argument("--incremental",    action="store_true",
                         help="Only embed frames not already in the index and append them")
    p_build.add_argument("--batch_size",      type=int, default=32)
    p_build.add_argument("--with_masks",      action="store_true",
                         help="Enable Stage B Mask Inversion (OWLv2 + SAM)")
    p_build.add_argument("--sam_checkpoint",  default=SAM_CHECKPOINT_DEFAULT,
                         help=f"Path to SAM .pth checkpoint (default: {SAM_CHECKPOINT_DEFAULT})")
    p_build.add_argument("--regen_masks",          action="store_true",
                         help="Regenerate masks even if captions.pt already exists")
    p_build.add_argument("--max_masks_per_frame",  type=int, default=5,
                         help="Max object masks to process per frame in MI mode (default: 5)")

    p_query = sub.add_parser("query")
    add_common(p_query)
    p_query.add_argument("--query",      nargs="+", default=None,
                         help="Query image path(s) (mutually exclusive with --query_text)")
    p_query.add_argument("--query_text", default=None, metavar="TEXT",
                         help="Text description to retrieve by (clip/siglip only; "
                              "mutually exclusive with --query)")
    p_query.add_argument("--top_k",      type=int, default=10)
    p_query.add_argument("--output",     default=os.path.expanduser(
        "~/Projects/robocasa_data/retrieval_result.png"))
    p_query.add_argument("--show",       action="store_true")
    p_query.add_argument("--with_masks",     dest="with_masks", action="store_true",  default=None,
                         help="Use MaO index (retrieval_index_{model}_mao)")
    p_query.add_argument("--no_masks",       dest="with_masks", action="store_false",
                         help="Use plain index (retrieval_index_{model})")
    p_query.add_argument("--rerank",         action="store_true",
                         help="Rerank top-K results with Gemini multimodal")
    p_query.add_argument("--skip_rerank",   action="store_true",
                         help="Skip all Gemini reranking/verification (for large pools)")
    p_query.add_argument("--output_pose",    default=None, metavar="PATH",
                         help="Save top-1 frame path + robot XY to JSON (for navigation)")
    p_query.add_argument("--gemini_model",   default="gemini-2.5-flash",
                         help="Gemini model to use for reranking (default: gemini-2.5-flash)")
    p_query.add_argument("--gemini_api_key", default="AIzaSyCdE4FkuAS0h6EtAvLSCXAXpKq6bpo-uB4",
                         help="Gemini API key (default: GOOGLE_API_KEY env var)")

    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
        sys.exit(1)

    if args.subcommand == "build":
        cmd_build(args)
    elif args.subcommand == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
