"""Pick SANPO-Real frames that have an OBSTACLE close to the camera, and download them for evaluation.

Why
    Random frames (one per second) rarely contain anything within 2-3 m, but those are exactly the
    frames where a warning matters. This script finds them with SANPO's own labels:
    an "obstacle" pixel (pedestrian, vehicle, pole, ... = sanpo.OBSTACLE_CLASSES) whose ground-truth
    depth is below --near-m meters.

Steps (per session, in random order with --seed)
    1. List frames that have RGB, dense depth AND a segmentation mask (masks exist only on labeled frames).
    2. Take --candidates evenly spaced frames and download their mask + depth (small files).
    3. Count near-obstacle pixels in each candidate; keep the --per-session frames with the most
       (at least --min-px pixels).
    4. Download the RGB frames of the kept ones. Stop after --sessions sessions with kept frames.

The output folder has the same layout as download_sanpo.py, so eval_sanpo.py can evaluate it directly
(only frames with an RGB image are indexed, so the extra candidate files are ignored).

Run from the repo root:
    python fast_path/depth/scripts/select_near_sanpo.py --root data/sanpo_near --sessions 10 --per-session 5
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fast_path.depth import sanpo  # noqa: E402


def near_obstacle_pixels(depth_path: Path, mask_path: Path, near_m: float) -> int:
    """Number of obstacle pixels closer than near_m meters (mask resized to the depth map's size)."""
    depth = sanpo.read_depth(depth_path)
    semantic = sanpo.read_semantic(mask_path)
    if semantic.shape != depth.shape:
        semantic = cv2.resize(semantic, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    near = np.isin(semantic, sanpo.OBSTACLE_IDS) & np.isfinite(depth) & (depth > 0.1) & (depth < near_m)
    return int(near.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/sanpo_near", help="local output folder")
    ap.add_argument("--split", default="sanpo-real", choices=["sanpo-real"])
    ap.add_argument("--camera", default="chest", choices=["chest", "head"])
    ap.add_argument("--sessions", type=int, default=10, help="sessions to keep frames from")
    ap.add_argument("--per-session", type=int, default=5, help="frames kept per session (most near-obstacle pixels)")
    ap.add_argument("--candidates", type=int, default=30, help="evenly spaced labeled frames checked per session")
    ap.add_argument("--near-m", type=float, default=3.0, help="an obstacle closer than this counts as near")
    ap.add_argument("--min-px", type=int, default=2000,
                    help="minimum near-obstacle pixels in the 1280x720 depth map (2000 px ~ a 45x45 patch)")
    ap.add_argument("--max-scan", type=int, default=150, help="give up after checking this many sessions")
    ap.add_argument("--official-split", choices=["train", "test"], default=None)
    ap.add_argument("--exclude", default="", help="comma-separated session ids to skip (write --exclude=ID1,ID2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.root)
    sessions = sanpo.list_sessions(args.split)
    if args.official_split:
        official = sanpo.official_split_sessions(args.split, args.official_split)
        sessions = [s for s in sessions if s in official]
    excluded = {s for s in args.exclude.split(",") if s}
    sessions = [s for s in sessions if s not in excluded]
    random.Random(args.seed).shuffle(sessions)
    print(f"{len(sessions)} candidate sessions ({len(excluded)} excluded); "
          f"looking for obstacles within {args.near_m:g} m")

    kept_sessions, kept_frames, scanned = 0, 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        for session in sessions:
            if kept_sessions >= args.sessions or scanned >= args.max_scan:
                break
            scanned += 1
            if args.camera not in sanpo.session_cameras(args.split, session):
                continue
            masks = set(sanpo.list_frame_ids(args.split, session, args.camera, "segmentation_masks"))
            if not masks:
                continue  # this session has no labels
            depths = set(sanpo.list_frame_ids(args.split, session, args.camera, "depth_maps"))
            rgbs = set(sanpo.list_frame_ids(args.split, session, args.camera, "video_frames"))
            usable = sorted(masks & depths & rgbs)
            if not usable:
                continue
            step = max(1, len(usable) // args.candidates)
            candidates = usable[::step][: args.candidates]

            # Step 2: masks + depth of the candidates
            base = f"{args.split}/{session}/camera_{args.camera}/left"
            jobs = [f"{base}/depth_maps/{fid}.float16.gz" for fid in candidates]
            jobs += [f"{base}/segmentation_masks/{fid}.png" for fid in candidates]
            list(pool.map(lambda job: sanpo.download(job, root), jobs))

            # Step 3: score and keep the frames with the most near-obstacle pixels
            left = root / base
            scores = list(pool.map(lambda fid: (near_obstacle_pixels(left / "depth_maps" / f"{fid}.float16.gz",
                                                                     left / "segmentation_masks" / f"{fid}.png",
                                                                     args.near_m), fid), candidates))
            good = sorted((s for s in scores if s[0] >= args.min_px), reverse=True)
            picked = sorted(fid for _, fid in good[: args.per_session])
            print(f"  {session}: {len(candidates)} labeled candidates, {len(good)} with an obstacle "
                  f"within {args.near_m:g} m -> taking {len(picked)}", flush=True)
            if not picked:
                continue

            # Step 4: RGB of the kept frames (+ the session description, which holds the camera intrinsics)
            jobs = [f"{base}/video_frames/{fid}.png" for fid in picked] + [f"{args.split}/{session}/description.json"]
            list(pool.map(lambda job: sanpo.download(job, root), jobs))
            kept_sessions += 1
            kept_frames += len(picked)

    print(f"Done: {kept_frames} frames from {kept_sessions} sessions (checked {scanned}) in {root.resolve()}")
    if kept_frames == 0:
        sys.exit("No frame had an obstacle close enough. Try a larger --near-m or a smaller --min-px.")


if __name__ == "__main__":
    main()
