"""SANPO dataset (Google Research) -- file formats, remote listing/download, local index.

The dataset lives in the public GCS bucket gs://gresearch/sanpo_dataset/v0 and is
readable over plain HTTPS, so no gsutil / gcloud is needed.

Layout (same locally, under --root):
    sanpo-real/<session>/description.json
    sanpo-real/<session>/camera_chest/left/video_frames/000000.png          2208x1242 RGB, 15 fps
    sanpo-real/<session>/camera_chest/left/depth_maps/000000.float16.gz     dense, 1280x720, meters
    sanpo-real/<session>/camera_chest/left/zed_depth_maps/000000.float16.gz sparse ZED SDK depth, full res, 0 = invalid
    sanpo-real/<session>/camera_chest/left/segmentation_masks/000000.png    only on annotated frames
    sanpo-synthetic/<session>/camera_head/left/...                          same names, dense GT, 5 fps

Depth file format: gzip of float16 values; the first two values are (H, W), the rest is HxW depth in meters.
Segmentation PNG: R channel = semantic class id (see LABELS), B channel = instance id.
"""

from __future__ import annotations

import gzip
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

BUCKET = "gresearch"
REMOTE_ROOT = "sanpo_dataset/v0"
PUBLIC_URL = f"https://storage.googleapis.com/{BUCKET}"
LIST_API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
SPLITS = ("sanpo-real", "sanpo-synthetic")
GT_KINDS = ("depth_maps", "zed_depth_maps")

# sanpo_dataset/v0/labelmap.json
LABELS = [
    "unlabeled", "road", "curb", "sidewalk", "guard rail/road barrier", "crosswalk", "paved trail",
    "building", "wall/fence", "hand rail", "opening-door", "opening-gate", "pedestrian", "rider",
    "animal", "stairs", "water body", "other walkable surface", "inaccessible surface",
    "railway track", "obstacle", "vehicle", "traffic sign", "traffic light", "pole", "bus stop",
    "bike rack", "sky", "tree", "vegetation", "terrain",
]
# Things a walking person can collide with -- used for the "obstacle pixels" metric.
OBSTACLE_CLASSES = [
    "guard rail/road barrier", "pedestrian", "rider", "animal", "obstacle", "vehicle",
    "traffic sign", "traffic light", "pole", "bus stop", "bike rack", "tree",
]
OBSTACLE_IDS = np.array([LABELS.index(c) for c in OBSTACLE_CLASSES])


# --------------------------------------------------------------------------- file formats

def read_depth(path: str | Path) -> np.ndarray:
    """Load a SANPO *.float16.gz depth map -> float32 HxW meters (0 / nan = invalid)."""
    with gzip.open(path, "rb") as f:
        a = np.frombuffer(f.read(), dtype=np.float16)
    h, w = int(a[0]), int(a[1])
    if a.size - 2 != h * w:
        raise ValueError(f"{path}: header says {h}x{w} but has {a.size - 2} values")
    return a[2:].reshape(h, w).astype(np.float32)


def read_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_semantic(path: str | Path) -> np.ndarray:
    """Semantic class ids (HxW uint8) from a SANPO segmentation PNG."""
    return np.asarray(Image.open(path).convert("RGB"))[..., 0]


# --------------------------------------------------------------------------- remote (GCS over HTTPS)

def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def list_remote(prefix: str, dirs_only: bool = False) -> tuple[list[str], list[str]]:
    """List one 'directory' level of the bucket. Returns (sub_prefixes, object_names)."""
    prefixes, names, token = [], [], None
    while True:
        q = {"prefix": prefix, "delimiter": "/", "maxResults": "1000",
             "fields": "prefixes,nextPageToken" + ("" if dirs_only else ",items(name)")}
        if token:
            q["pageToken"] = token
        d = _get_json(f"{LIST_API}?{urllib.parse.urlencode(q)}")
        prefixes += d.get("prefixes", [])
        names += [i["name"] for i in d.get("items", []) if not i["name"].endswith("$folder$")]
        token = d.get("nextPageToken")
        if not token:
            return prefixes, names


def list_sessions(split: str) -> list[str]:
    prefixes, _ = list_remote(f"{REMOTE_ROOT}/{split}/", dirs_only=True)
    return sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes)


def list_frame_ids(split: str, session: str, camera: str, folder: str) -> list[str]:
    """Frame ids ('000123') present in camera_<camera>/left/<folder> of a session."""
    _, names = list_remote(f"{REMOTE_ROOT}/{split}/{session}/camera_{camera}/left/{folder}/")
    return sorted(n.rsplit("/", 1)[-1].split(".", 1)[0] for n in names)


def session_cameras(split: str, session: str) -> list[str]:
    prefixes, _ = list_remote(f"{REMOTE_ROOT}/{split}/{session}/", dirs_only=True)
    return [p.rstrip("/").rsplit("/", 1)[-1].removeprefix("camera_") for p in prefixes
            if p.rstrip("/").rsplit("/", 1)[-1].startswith("camera_")]


def download(rel_path: str, root: Path) -> Path:
    """Download sanpo_dataset/v0/<rel_path> to root/<rel_path> (skips existing files)."""
    dst = root / rel_path
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{PUBLIC_URL}/{urllib.parse.quote(f'{REMOTE_ROOT}/{rel_path}')}"
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dst)
    return dst


# --------------------------------------------------------------------------- local index

@dataclass
class SanpoFrame:
    split: str
    session: str
    camera: str
    frame_id: str
    rgb: Path
    depth: Path | None  # dense depth_maps
    zed_depth: Path | None  # sparse zed_depth_maps (real only)
    seg: Path | None

    @property
    def key(self) -> str:
        return f"{self.split}/{self.session}/{self.camera}/{self.frame_id}"

    def gt_path(self, gt: str) -> Path | None:
        return self.depth if gt == "depth_maps" else self.zed_depth


def index_local(root: str | Path, split: str | None = None, camera: str | None = None,
                gt: str = "depth_maps") -> list[SanpoFrame]:
    """All downloaded frames under root that have an RGB image and the requested GT."""
    root = Path(root)
    frames = []
    for rgb in sorted(root.glob("sanpo-*/*/camera_*/left/video_frames/*.png")):
        left = rgb.parent.parent
        sp, sess, cam = left.parts[-4], left.parts[-3], left.parts[-2].removeprefix("camera_")
        if (split and sp != split) or (camera and cam != camera):
            continue
        fid = rgb.stem

        def opt(folder: str, ext: str) -> Path | None:
            p = left / folder / f"{fid}{ext}"
            return p if p.exists() else None

        fr = SanpoFrame(sp, sess, cam, fid, rgb, opt("depth_maps", ".float16.gz"),
                        opt("zed_depth_maps", ".float16.gz"), opt("segmentation_masks", ".png"))
        if fr.gt_path(gt) is not None:
            frames.append(fr)
    return frames


class SanpoDepthDataset:
    """Minimal torch-style dataset over local frames, for fine-tuning / distillation later.

    Returns dict(rgb=HxWx3 uint8, depth=hxw float32 meters, valid=hxw bool, key=str).
    RGB is resized to the GT resolution so pixels correspond 1:1.
    """

    def __init__(self, frames: list[SanpoFrame], gt: str = "depth_maps", max_depth: float = 80.0):
        self.frames, self.gt, self.max_depth = frames, gt, max_depth

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> dict:
        fr = self.frames[i]
        depth = read_depth(fr.gt_path(self.gt))
        img = Image.open(fr.rgb).convert("RGB")
        if img.size != (depth.shape[1], depth.shape[0]):
            img = img.resize((depth.shape[1], depth.shape[0]), Image.BILINEAR)
        valid = np.isfinite(depth) & (depth > 0) & (depth <= self.max_depth)
        return {"rgb": np.asarray(img), "depth": depth, "valid": valid, "key": fr.key}
