import os
import shutil
import random
import argparse


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
    """nh33_hazy.png -> nh33.png"""
    stem, ext = os.path.splitext(hazy_fname)
    if stem.endswith("_hazy"):
        stem = stem[: -len("_hazy")]
    return stem + ext.lower()


def build_pairs(src_hazy_dir, src_clear_dir):
    clear_set = set(list_images(src_clear_dir))
    pairs = []
    for hazy_fname in list_images(src_hazy_dir):
        clear_fname = hazy_to_clear_name(hazy_fname)
        if clear_fname in clear_set:
            pairs.append((hazy_fname, clear_fname))
        else:
            print(f"[warn] no clear for hazy: {hazy_fname} (expected {clear_fname})")
    return pairs


def copy_pair(src_hazy_dir, src_clear_dir, hazy_fname, clear_fname, dst_hazy_dir, dst_clear_dir):
    src_hazy = os.path.join(src_hazy_dir, hazy_fname)
    src_clear = os.path.join(src_clear_dir, clear_fname)

    if not os.path.exists(src_hazy):
        raise FileNotFoundError(f"Missing hazy file: {src_hazy}")
    if not os.path.exists(src_clear):
        raise FileNotFoundError(f"Missing clear file: {src_clear}")

    ensure_dir(dst_hazy_dir)
    ensure_dir(dst_clear_dir)

    dst_hazy = os.path.join(dst_hazy_dir, hazy_fname)
    dst_clear = os.path.join(dst_clear_dir, clear_fname)

    if os.path.abspath(src_hazy) != os.path.abspath(dst_hazy):
        shutil.copy2(src_hazy, dst_hazy)
    if os.path.abspath(src_clear) != os.path.abspath(dst_clear):
        shutil.copy2(src_clear, dst_clear)


def split(dataset, n_train, n_val, n_test, seed=42, source_split="train", keep_existing_test=True):
    """
    dataset example: real_raw/NH-Haze
    Expects source at: data/<dataset>/<source_split>/{hazy,clear}
    Writes to:         data/<dataset>/{train,val,test}/{hazy,clear}

    NH-Haze naming: hazy = nh33_hazy.png, clear = nh33.png
    """
    root = os.path.join("data", dataset)
    src_hazy_dir = os.path.join(root, source_split, "hazy")
    src_clear_dir = os.path.join(root, source_split, "clear")

    if not os.path.isdir(src_hazy_dir) or not os.path.isdir(src_clear_dir):
        raise FileNotFoundError(
            f"Source folders not found:\n  {src_hazy_dir}\n  {src_clear_dir}"
        )

    pairs = build_pairs(src_hazy_dir, src_clear_dir)
    if len(pairs) == 0:
        raise ValueError(
            f"No hazy/clear pairs found under {src_hazy_dir} / {src_clear_dir}"
        )

    total = len(pairs)
    req = n_train + n_val + n_test
    if req > total:
        raise ValueError(
            f"Requested {req} samples (train+val+test), but only {total} paired samples available."
        )

    random.seed(seed)
    random.shuffle(pairs)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val : n_train + n_val + n_test]

    # If source_split is train, avoid self-copy by snapshotting source first
    tmp_src = None
    if source_split == "train":
        tmp_src = os.path.join(root, "_tmp_split_source")
        if os.path.isdir(tmp_src):
            shutil.rmtree(tmp_src)
        shutil.copytree(os.path.join(root, "train"), tmp_src)
        src_hazy_dir = os.path.join(tmp_src, "hazy")
        src_clear_dir = os.path.join(tmp_src, "clear")

    # Recreate train/val always
    for split_name in ["train", "val"]:
        clear_dir(os.path.join(root, split_name, "hazy"))
        clear_dir(os.path.join(root, split_name, "clear"))

    # Recreate test only when splitting into test and not preserving existing test
    if n_test > 0 and not keep_existing_test:
        clear_dir(os.path.join(root, "test", "hazy"))
        clear_dir(os.path.join(root, "test", "clear"))
    else:
        ensure_dir(os.path.join(root, "test", "hazy"))
        ensure_dir(os.path.join(root, "test", "clear"))

    for hazy_f, clear_f in train_pairs:
        copy_pair(
            src_hazy_dir,
            src_clear_dir,
            hazy_f,
            clear_f,
            os.path.join(root, "train", "hazy"),
            os.path.join(root, "train", "clear"),
        )

    for hazy_f, clear_f in val_pairs:
        copy_pair(
            src_hazy_dir,
            src_clear_dir,
            hazy_f,
            clear_f,
            os.path.join(root, "val", "hazy"),
            os.path.join(root, "val", "clear"),
        )

    if n_test > 0:
        for hazy_f, clear_f in test_pairs:
            copy_pair(
                src_hazy_dir,
                src_clear_dir,
                hazy_f,
                clear_f,
                os.path.join(root, "test", "hazy"),
                os.path.join(root, "test", "clear"),
            )

    if tmp_src and os.path.isdir(tmp_src):
        shutil.rmtree(tmp_src)

    print(f"[ok] Dataset: {dataset}")
    print(f"[ok] Paired source images: {total}")
    print(
        f"[ok] Split -> train: {len(train_pairs)} | val: {len(val_pairs)} | test: {len(test_pairs)}"
    )
    print(f"[ok] Output root: {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. real_raw/NH-Haze")
    ap.add_argument("--train", type=int, required=True)
    ap.add_argument("--val", type=int, required=True)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source_split", default="train", choices=["train", "all"])
    ap.add_argument(
        "--keep_existing_test",
        action="store_true",
        help="If set, do not overwrite existing test when --test 0.",
    )
    args = ap.parse_args()

    split(
        dataset=args.dataset,
        n_train=args.train,
        n_val=args.val,
        n_test=args.test,
        seed=args.seed,
        source_split=args.source_split,
        keep_existing_test=args.keep_existing_test,
    )