"""Download the SANPO dataset from the Modal Volume to this PC, in one script.

Downloading ~72k small files one by one is slow (about one request per file),
so this script works in two stages:

  Stage 1 (on Modal): pack the dataset into ~2 GB .tar archives on the Volume.
      Plain .tar (bundling only, no compression; JPG/PNG are already compressed).
      Several archives are packed in parallel containers.
  Stage 2 (on this PC): download the archives (several at once), unpack each
      into LOCAL_ROOT, then delete the .tar to save disk space.
  Stage 3 (on Modal): when everything is unpacked, delete the archives from the
      Volume (the dataset itself stays there).

Result: fast_path/object_detection/data/sanpo/<DATASET>/frames, masks, index, views, sanpo_meta
(git-ignored). About 20 GB.

Resumable at every step: the packing plan (manifest) is made once and reused;
packed archives are not packed again; unpacked archives have a ".done" marker
and are skipped; downloads and archives are written to .tmp files first, so
Ctrl+C or a crash never leaves broken files. Just run it again to continue.
If the dataset on the Volume changes (e.g. a new view), delete the folder
_archives/<DATASET> on the Volume and run again.

Run from the repo root (PowerShell):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session1.toml"
    python fast_path/object_detection/data_code/sanpo/download_from_volume.py
"""

from __future__ import annotations

import modal

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
VOLUME_NAME = "sanpo"
DATASET = "sanpo_real_chest_5fps"  # folder inside the Volume
ARCHIVE_SIZE_GB = 2.0  # target size of each .tar archive (changing it re-packs everything)
PACK_CONTAINERS = 10  # archives packed in parallel on Modal
PARALLEL_DOWNLOADS = 10  # archives downloaded at the same time on this PC
SPEED_WINDOW_S = 30  # speed/ETA use the last N seconds (not the whole-run average)
READ_THREADS = 32  # files read at the same time inside each packing container
READ_BATCH = 500  # files per batch (limits memory use)
PRINT_EVERY_S = 2  # download progress line every N seconds
KEEP_TARS = False  # True = keep .tar files after unpacking (needs ~2x disk space)
DELETE_ARCHIVES_WHEN_DONE = True  # delete the .tar archives from the Volume after a complete download
# =============================================================================

DATA_MOUNT = "/data"
ARCHIVE_DIR = f"_archives/{DATASET}"

app = modal.App("blind-guide-sanpo-download-local")
volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.12")


def fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


# ============================================================================= Stage 1: pack on Modal

@app.function(image=image, volumes={DATA_MOUNT: volume}, timeout=3600)
def make_plan() -> dict:
    """List all dataset files and group them into ~ARCHIVE_SIZE_GB parts (reuses an existing plan)."""
    import json
    import os
    from pathlib import Path

    root = Path(DATA_MOUNT)
    manifest_path = root / ARCHIVE_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("archive_size_gb", ARCHIVE_SIZE_GB) == ARCHIVE_SIZE_GB:
            print("Reusing existing packing plan:", manifest_path, flush=True)
            return manifest
        # Archive size changed: old archives no longer match the plan, so remove them and re-plan.
        import shutil

        print(f"Archive size changed ({manifest.get('archive_size_gb')} -> {ARCHIVE_SIZE_GB} GB): "
              f"deleting old archives and re-planning ...", flush=True)
        shutil.rmtree(root / ARCHIVE_DIR)
        volume.commit()

    print(f"Listing files in {root / DATASET} ...", flush=True)
    files = []
    for dirpath, _, names in os.walk(root / DATASET):
        for name in names:
            if not name.endswith(".tmp"):
                p = Path(dirpath) / name
                files.append([str(p.relative_to(root)).replace("\\", "/"), p.stat().st_size])
    files.sort()

    limit, parts, current, current_size = ARCHIVE_SIZE_GB * 1e9, [], [], 0
    for path, size in files:
        if current and current_size + size > limit:
            parts.append(current)
            current, current_size = [], 0
        current.append([path, size])
        current_size += size
    if current:
        parts.append(current)

    manifest = {
        "dataset": DATASET, "archive_size_gb": ARCHIVE_SIZE_GB, "plan_id": f"{ARCHIVE_SIZE_GB}gb_{len(parts)}parts",
        "total_files": len(files), "total_bytes": sum(s for _, s in files),
        "parts": [{"name": f"part_{i:03d}.tar", "files": p, "bytes": sum(s for _, s in p)} for i, p in enumerate(parts)],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_name("manifest.json.tmp")
    tmp.write_text(json.dumps(manifest))
    os.replace(tmp, manifest_path)
    volume.commit()
    print(f"Plan: {len(files):,} files -> {len(parts)} archives", flush=True)
    return manifest


@app.function(image=image, volumes={DATA_MOUNT: volume}, timeout=3 * 3600, max_containers=PACK_CONTAINERS,
              retries=2, cpu=4, memory=4096)
def pack_part(name: str, files: list) -> dict:
    """Write one .tar archive on the Volume (skipped if it already exists).

    Speed: reading files from the Volume one by one is slow (network delay per
    file), so READ_THREADS files are read at the same time, in batches; the
    archive is still written in order.
    """
    import io
    import os
    import tarfile
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    root = Path(DATA_MOUNT)
    target = root / ARCHIVE_DIR / name
    if target.exists():
        return {"name": name, "bytes": target.stat().st_size, "skipped": True}

    def read(path: str) -> bytes:
        return (root / path).read_bytes()

    tmp = target.with_name(name + ".tmp")
    start, done, done_bytes = time.time(), 0, 0
    with tarfile.open(tmp, "w") as tar, ThreadPoolExecutor(READ_THREADS) as pool:
        for b in range(0, len(files), READ_BATCH):
            batch = [path for path, _ in files[b:b + READ_BATCH]]
            for path, data in zip(batch, pool.map(read, batch)):  # parallel reads, ordered results
                info = tarfile.TarInfo(path)  # path relative to the Volume root
                info.size, info.mtime = len(data), int(time.time())
                tar.addfile(info, io.BytesIO(data))
                done_bytes += len(data)
            done += len(batch)
            elapsed = time.time() - start
            print(f"pack {name}: {done:,}/{len(files):,} files | {done_bytes / 1e9:.2f} GB | "
                  f"{done_bytes / 1e6 / max(elapsed, 1e-9):.0f} MB/s | elapsed {fmt_time(elapsed)} | "
                  f"ETA {fmt_time(elapsed / done * (len(files) - done))}", flush=True)
    os.replace(tmp, target)
    volume.commit()
    return {"name": name, "bytes": target.stat().st_size, "skipped": False}


def stage1_pack() -> dict:
    """Run the packing on Modal and return the manifest (with each archive's size)."""
    import time

    print(f"STAGE 1/2: packing '{VOLUME_NAME}/{DATASET}' into ~{ARCHIVE_SIZE_GB} GB archives on Modal ...", flush=True)
    manifest = make_plan.remote()
    parts = manifest["parts"]
    print(f"{manifest['total_files']:,} files, {manifest['total_bytes'] / 1e9:.2f} GB -> {len(parts)} archives "
          f"({PACK_CONTAINERS} packed in parallel)", flush=True)

    start, new, failed = time.time(), 0, []
    jobs = [(p["name"], p["files"]) for p in parts]
    sizes = {}
    for i, r in enumerate(pack_part.starmap(jobs, order_outputs=False, return_exceptions=True), 1):
        if isinstance(r, Exception):
            failed.append(r)
            print(f"[pack {i}/{len(parts)}] FAILED: {r!r}", flush=True)
            continue
        sizes[r["name"]] = r["bytes"]
        new += 0 if r["skipped"] else 1
        elapsed = time.time() - start
        eta = elapsed / new * (len(parts) - i) if new else 0
        print(f"[pack {i}/{len(parts)}] {r['name']}: {r['bytes'] / 1e9:.2f} GB"
              f"{' (already packed)' if r['skipped'] else ''} | elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)}",
              flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} archive(s) failed to pack. Run the script again to retry.")
    for p in parts:
        p["tar_bytes"] = sizes[p["name"]]
    return manifest


# ============================================================================= Stage 2: download on this PC

def stage2_download(manifest: dict) -> bool:
    """Returns True when every archive is downloaded and unpacked."""
    import os
    import tarfile
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    local_root = Path(__file__).resolve().parents[2] / "data" / "sanpo"  # fast_path/object_detection/data/sanpo
    archive_dir = local_root / "_archives" / DATASET
    archive_dir.mkdir(parents=True, exist_ok=True)

    parts = manifest["parts"]
    todo = [p for p in parts if not (archive_dir / f"{p['name']}.done").exists()]
    print(f"STAGE 2/2: download + unpack | archives already done: {len(parts) - len(todo)} | to do: {len(todo)}",
          flush=True)
    print(f"Destination: {local_root / DATASET}", flush=True)
    if not todo:
        print("Nothing to do.")
        return True

    total = sum(p["tar_bytes"] for p in todo)
    state = {"done": 0, "last": time.time()}
    lock, start = threading.Lock(), time.time()
    history = [(start, 0)]  # (time, bytes done) samples for the recent speed

    def report(force: bool = False) -> None:
        now = time.time()
        elapsed = now - start
        history.append((now, state["done"]))
        while len(history) > 2 and now - history[0][0] > SPEED_WINDOW_S:
            history.pop(0)
        t0, b0 = history[0]
        speed = (state["done"] - b0) / (now - t0) if now > t0 else 0  # speed over the last SPEED_WINDOW_S
        eta = (total - state["done"]) / speed if speed else 0
        print(f"download {state['done'] / 1e9:.2f}/{total / 1e9:.2f} GB ({100 * state['done'] / total:.1f}%) | "
              f"{speed / 1e6:.1f} MB/s | elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)}", flush=True)

    def add_bytes(n: int) -> None:
        with lock:
            state["done"] += n
            if time.time() - state["last"] >= PRINT_EVERY_S:
                state["last"] = time.time()
                report()

    def process(part: dict) -> str:
        name, tar_path = part["name"], archive_dir / part["name"]
        if tar_path.exists() and tar_path.stat().st_size == part["tar_bytes"]:
            add_bytes(part["tar_bytes"])  # downloaded in an earlier run, not unpacked yet
        else:
            tmp = archive_dir / f"{name}.tmp"
            with open(tmp, "wb") as f:
                for chunk in volume.read_file(f"{ARCHIVE_DIR}/{name}"):
                    f.write(chunk)
                    add_bytes(len(chunk))
            os.replace(tmp, tar_path)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(local_root, filter="data")  # "data" filter blocks unsafe paths
        (archive_dir / f"{name}.done").write_text(f"{len(part['files'])} files\n")
        if not KEEP_TARS:
            tar_path.unlink()
        return f"{name}: unpacked {len(part['files']):,} files"

    failed = []
    with ThreadPoolExecutor(PARALLEL_DOWNLOADS) as pool:
        futures = {pool.submit(process, p): p["name"] for p in todo}
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    print(f"[unpack {i}/{len(todo)}] {fut.result()}", flush=True)
                except Exception as exc:
                    failed.append(futures[fut])
                    print(f"[unpack {i}/{len(todo)}] FAILED {futures[fut]}: {exc!r}", flush=True)
                report()
        except KeyboardInterrupt:
            print("\nStopped by user. Finished archives are kept; run the same command to continue.", flush=True)
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    print(f"STAGE 2 DONE in {fmt_time(time.time() - start)}"
          f"{f' | {len(failed)} failed: run again to retry' if failed else ''}")
    return not failed


if __name__ == "__main__":
    import time

    t0 = time.time()
    with modal.enable_output():  # shows logs from the Modal containers
        with app.run():
            manifest = stage1_pack()
    complete = stage2_download(manifest)

    # Stage 3: delete the archives from the Volume once everything is unpacked locally
    # (the dataset itself stays on the Volume; only the extra .tar copies are removed).
    if complete and DELETE_ARCHIVES_WHEN_DONE:
        print(f"STAGE 3: deleting '{VOLUME_NAME}/{ARCHIVE_DIR}' (archives no longer needed) ...", flush=True)
        volume.remove_file(ARCHIVE_DIR, recursive=True)
        print("Archives deleted from the Volume.", flush=True)
    elif not complete:
        print("Archives kept on the Volume because some parts failed. Run again to finish.", flush=True)
    print(f"ALL DONE in {fmt_time(time.time() - t0)}")
