import argparse
import os

import yaml


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    return sorted(
        f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def infer_clear_name(hazy_fname, clear_set):
    stem, hazy_ext = os.path.splitext(hazy_fname)
    candidates = []

    # Exact same filename.
    candidates.append(hazy_fname)

    # NH-Haze style: nh33_hazy.png -> nh33.png
    if stem.endswith("_hazy"):
        candidates.append(stem[: -len("_hazy")] + hazy_ext.lower())

    # SOTS style: 1400_1.png -> 1400.png, 0001_0.8_0.2.jpg -> 0001.png
    candidates.append(stem.split("_")[0] + ".png")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in clear_set:
            return candidate
    return None


def verify_split(split, ds):
    lq_root = ds["dataroot_LQ"]
    gt_root = ds["dataroot_GT"]

    print(f"\n[{split}]")
    print(f"  hazy : {lq_root}")
    print(f"  clear: {gt_root}")

    if not os.path.isdir(lq_root):
        print("  status: missing hazy folder")
        return False
    if not os.path.isdir(gt_root):
        print("  status: missing clear folder")
        return False

    hazy_files = list_images(lq_root)
    clear_files = list_images(gt_root)
    clear_set = set(clear_files)

    matched = 0
    unmatched = []
    reused_clear = {}

    for hazy_fname in hazy_files:
        clear_name = infer_clear_name(hazy_fname, clear_set)
        if clear_name is None:
            unmatched.append(hazy_fname)
            continue
        matched += 1
        reused_clear[clear_name] = reused_clear.get(clear_name, 0) + 1

    multi_hazy_clear = sum(1 for v in reused_clear.values() if v > 1)

    print(f"  hazy_count          : {len(hazy_files)}")
    print(f"  clear_count         : {len(clear_files)}")
    print(f"  matched_hazy        : {matched}")
    print(f"  unmatched_hazy      : {len(unmatched)}")
    print(f"  reused_clear_images : {multi_hazy_clear}")

    if unmatched:
        print("  sample_unmatched:")
        for fname in unmatched[:10]:
            print(f"    - {fname}")

    ok = matched == len(hazy_files) and len(hazy_files) > 0 and len(clear_files) > 0
    print(f"  status: {'ok' if ok else 'needs attention'}")
    return ok


def verify(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    print(f"Config: {config_path}")
    print(f"Dataset name: {opt.get('name', 'unknown')}")

    ok = True
    for split, ds in opt["datasets"].items():
        split_ok = verify_split(split, ds)
        ok = ok and split_ok

    print(f"\nFinal status: {'ok' if ok else 'failed'}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.config) else 1)
