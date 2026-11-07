"""Download a manageable subset of SANPO (the full dataset is several TB).

Examples:
    # 5 real sessions, chest camera, 1 frame per second, 10 frames each (~200 MB)
    python scripts/download_sanpo.py --split sanpo-real --sessions 5

    # synthetic data (dense, perfect GT; head camera only)
    python scripts/download_sanpo.py --split sanpo-synthetic --camera head --sessions 5 --stride 5

    # just list sessions
    python scripts/download_sanpo.py --split sanpo-real --list
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

from depth import sanpo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/sanpo", help="local output folder")
    ap.add_argument("--split", default="sanpo-real", choices=sanpo.SPLITS)
    ap.add_argument("--camera", default="chest", choices=["chest", "head"],
                    help="chest matches our device placement (real only; synthetic has head only)")
    ap.add_argument("--sessions", type=int, default=5, help="number of sessions to sample")
    ap.add_argument("--session-ids", default="",
                    help="comma-separated session ids, overrides --sessions. Write it as "
                         "--session-ids=ID1,ID2 (the '=' matters: some ids start with '-')")
    ap.add_argument("--stride", type=int, default=15, help="take every Nth frame (real is 15 fps)")
    ap.add_argument("--max-frames", type=int, default=10, help="frames per session")
    ap.add_argument("--gt", nargs="+", default=["depth_maps"],
                    choices=["depth_maps", "zed_depth_maps", "segmentation_masks"],
                    help="what to fetch besides RGB (segmentation is only present on some frames)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="list sessions and exit")
    args = ap.parse_args()

    root = Path(args.root)
    sessions = sanpo.list_sessions(args.split)
    print(f"{args.split}: {len(sessions)} sessions in the bucket")
    if args.list:
        print("\n".join(sessions))
        return

    explicit = [s for s in args.session_ids.split(",") if s]
    if explicit:
        candidates = explicit
    else:
        candidates = sessions[:]
        random.Random(args.seed).shuffle(candidates)

    ext = {"depth_maps": ".float16.gz", "zed_depth_maps": ".float16.gz", "segmentation_masks": ".png"}
    jobs: list[str] = []
    chosen = 0
    for sess in candidates:
        if chosen >= (len(candidates) if explicit else args.sessions):
            break
        if args.camera not in sanpo.session_cameras(args.split, sess):
            continue
        base = f"{args.split}/{sess}/camera_{args.camera}/left"
        rgb_ids = set(sanpo.list_frame_ids(args.split, sess, args.camera, "video_frames"))
        have = {g: set(sanpo.list_frame_ids(args.split, sess, args.camera, g)) for g in args.gt}
        # require RGB + every requested depth GT; segmentation is optional (sparse annotations)
        usable = sorted(rgb_ids.intersection(*[v for g, v in have.items() if g != "segmentation_masks"]))
        picked = usable[:: args.stride][: args.max_frames]
        if not picked:
            print(f"  skip {sess}: no frames with {args.gt}")
            continue
        chosen += 1
        print(f"  {sess}: {len(usable)} usable frames -> taking {len(picked)}")
        jobs.append(f"{args.split}/{sess}/description.json")
        for fid in picked:
            jobs.append(f"{base}/video_frames/{fid}.png")
            for g in args.gt:
                if fid in have[g]:
                    jobs.append(f"{base}/{g}/{fid}{ext[g]}")

    if chosen == 0:
        sys.exit(f"No sessions with camera '{args.camera}' found in {args.split}.")

    errors = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(sanpo.download, j, root): j for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="download", unit="file"):
            try:
                fut.result()
            except Exception as e:  # keep going, report at the end
                errors += 1
                tqdm.write(f"FAILED {futs[fut]}: {e}")
    print(f"Done: {len(jobs) - errors}/{len(jobs)} files in {root.resolve()}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
