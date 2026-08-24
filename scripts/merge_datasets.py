"""将兼容的 YOLO 数据集合并为一个三类别数据集。"""

import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = PROJECT_ROOT / "public_datasets"
LOCAL_IMAGES = PROJECT_ROOT / "data" / "images"
LOCAL_LABELS = PROJECT_ROOT / "data" / "labels"
OUTPUT_ROOT = PROJECT_ROOT / "data_merged"

CLASS_NAMES = {0: "cup", 1: "book", 2: "pen"}

# 来源数据集类别编号 -> 项目最终类别编号。
PUBLIC_MAPPINGS = {
    "cup_dataset_2": {0: 0},
    "pen_book_dataset": {0: 1, 1: 2},
}

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def output_dirs(split: str) -> tuple[Path, Path]:
    image_dir = OUTPUT_ROOT / "images" / split
    label_dir = OUTPUT_ROOT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, label_dir


def convert_label(label_path: Path, mapping: dict[int, int]) -> list[str]:
    converted = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label format: {label_path}:{line_number}")

        source_id = int(parts[0])
        if source_id not in mapping:
            continue

        parts[0] = str(mapping[source_id])
        converted.append(" ".join(parts))

    return converted


def copy_sample(
    image_path: Path,
    label_path: Path,
    split: str,
    prefix: str,
    mapping: dict[int, int],
) -> bool:
    image_dir, label_dir = output_dirs(split)
    output_stem = f"{prefix}_{image_path.stem}"
    output_image = image_dir / f"{output_stem}{image_path.suffix.lower()}"
    output_label = label_dir / f"{output_stem}.txt"
    converted = convert_label(label_path, mapping)

    # 只保留至少包含一个目标物体的图片。
    if not converted:
        return False

    shutil.copy2(image_path, output_image)
    output_label.write_text("\n".join(converted) + "\n", encoding="utf-8")
    return True


def merge_public_dataset(dataset_name: str, mapping: dict[int, int]) -> int:
    dataset_root = PUBLIC_ROOT / dataset_name
    copied = 0

    for source_split, target_split in (("train", "train"), ("valid", "val"), ("test", "test")):
        image_dir = dataset_root / source_split / "images"
        label_dir = dataset_root / source_split / "labels"
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label: {label_path}")

            copied += copy_sample(image_path, label_path, target_split, dataset_name, mapping)

    return copied


def merge_local_dataset() -> int:
    images = sorted(
        path for path in LOCAL_IMAGES.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    random.Random(42).shuffle(images)

    train_end = int(len(images) * 0.7)
    val_end = train_end + int(len(images) * 0.2)
    split_files = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    copied = 0
    for split, files in split_files.items():
        for image_path in files:
            label_path = LOCAL_LABELS / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing local label: {label_path}")
            copied += copy_sample(image_path, label_path, split, "local", {0: 0, 1: 1, 2: 2})

    return copied


def write_dataset_yaml() -> None:
    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: cup
  1: book
  2: pen
"""
    (OUTPUT_ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")


def validate() -> None:
    for split in SPLITS:
        image_dir = OUTPUT_ROOT / "images" / split
        label_dir = OUTPUT_ROOT / "labels" / split
        images = [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
        labels = list(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise RuntimeError(f"Image/label mismatch in {split}")
        print(f"{split}: {len(images)} images, {len(labels)} labels")


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_ROOT}. Remove it manually before rebuilding."
        )

    OUTPUT_ROOT.mkdir(parents=True)

    for dataset_name, mapping in PUBLIC_MAPPINGS.items():
        count = merge_public_dataset(dataset_name, mapping)
        print(f"{dataset_name}: merged {count} images")

    local_count = merge_local_dataset()
    print(f"local dataset: merged {local_count} images")

    write_dataset_yaml()
    validate()
    print(f"Merged dataset: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
