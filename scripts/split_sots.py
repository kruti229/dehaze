import argparse
import os
import random
import shutil
from collections import defaultdict


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    return sorted(
        f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def hazy_to_clear_name(hazy_fname):
    stem, _ = os.path.splitext(hazy_fname)
    clear_stem = stem.split("_")[0]
    return clear_stem + ".png"


def build_groups(src_hazy_dir, src_clear_dir):
    clear_names = set(list_images(src_clear_dir))
    groups = defaultdict(list)
    missing = []

    for hazy_fname in list_images(src_hazy_dir):
        clear_fname = hazy_to_clear_name(hazy_fname)
        if clear_fname in clear_names:
            clear_id = os.path.splitext(clear_fname)[0]
            groups[clear_id].append((hazy_fname, clear_fname))
        else:
            missing.append((hazy_fname, clear_fname))

    for hazy_fname, clear_fname in missing[:20]:
        print(f"[warn] no clear for hazy: {hazy_fname} -> expected {clear_fname}")
    if len(missing) > 20:
        print(f"[warn] ... and {len(missing) - 20} more unmatched hazy files")

    return dict(sorted(groups.items())), missing


def copy_pair(src_hazy_dir, src_clear_dir, hazy_fname, clear_fname, dst_hazy_dir, dst_clear_dir):
    ensure_dir(dst_hazy_dir)
    ensure_dir(dst_clear_dir)

    shutil.copy2(
        os.path.join(src_hazy_dir, hazy_fname),
        os.path.join(dst_hazy_dir, hazy_fname),
    )
    shutil.copy2(
        os.path.join(src_clear_dir, clear_fname),
        os.path.join(dst_clear_dir, clear_fname),
    )


def copy_group(src_hazy_dir, src_clear_dir, group_pairs, dst_hazy_dir, dst_clear_dir):
    for hazy_fname, clear_fname in group_pairs:
        copy_pair(
            src_hazy_dir,
            src_clear_dir,
            hazy_fname,
            clear_fname,
            dst_hazy_dir,
            dst_clear_dir,
        )


def split_sots(src_hazy_dir, src_clear_dir, out_root, train_count, val_count, test_count, seed=42, dry_run=False):
    if not os.path.isdir(src_hazy_dir):
        raise FileNotFoundError(f"Missing hazy dir: {src_hazy_dir}")
    if not os.path.isdir(src_clear_dir):
        raise FileNotFoundError(f"Missing clear dir: {src_clear_dir}")

    groups, missing = build_groups(src_hazy_dir, src_clear_dir)
    if not groups:
        raise ValueError("No matched hazy/clear pairs found.")

    clear_ids = list(groups.keys())
    total_groups = len(clear_ids)
    total_hazy = sum(len(v) for v in groups.values())
    requested = train_count + val_count + test_count
    if requested > total_groups:
        raise ValueError(
            f"Requested {requested} groups but only found {total_groups} matched clear-id groups."
        )

    random.seed(seed)
    random.shuffle(clear_ids)

    train_ids = clear_ids[:train_count]
    val_ids = clear_ids[train_count : train_count + val_count]
    test_ids = clear_ids[train_count + val_count : train_count + val_count + test_count]

    if dry_run:
        print(f"[ok] matched hazy pairs: {total_hazy}")
        print(f"[ok] matched clear-id groups: {total_groups}")
        print(f"[ok] unmatched hazy files: {len(missing)}")
        print(
            f"[ok] planned groups -> train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}"
        )
        print(
            f"[ok] planned hazy images -> train: {sum(len(groups[k]) for k in train_ids)} | "
            f"val: {sum(len(groups[k]) for k in val_ids)} | test: {sum(len(groups[k]) for k in test_ids)}"
        )
        return

    for split_name in ["train", "val", "test"]:
        clear_dir(os.path.join(out_root, split_name, "hazy"))
        clear_dir(os.path.join(out_root, split_name, "clear"))

    for split_name, split_ids in [
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids),
    ]:
        for clear_id in split_ids:
            copy_group(
                src_hazy_dir,
                src_clear_dir,
                groups[clear_id],
                os.path.join(out_root, split_name, "hazy"),
                os.path.join(out_root, split_name, "clear"),
            )

    print(f"[ok] matched hazy pairs: {total_hazy}")
    print(f"[ok] matched clear-id groups: {total_groups}")
    print(
        f"[ok] split groups -> train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}"
    )
    print(
        f"[ok] split hazy images -> train: {sum(len(groups[k]) for k in train_ids)} | "
        f"val: {sum(len(groups[k]) for k in val_ids)} | test: {sum(len(groups[k]) for k in test_ids)}"
    )
    print(f"[ok] output root: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_hazy", required=True)
    parser.add_argument("--src_clear", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--val", type=int, required=True)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    split_sots(
        src_hazy_dir=args.src_hazy,
        src_clear_dir=args.src_clear,
        out_root=args.out_root,
        train_count=args.train,
        val_count=args.val,
        test_count=args.test,
        seed=args.seed,
        dry_run=args.dry_run,
    )
