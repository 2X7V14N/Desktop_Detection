"""将新的 Book 数据集加入现有三类别数据集。"""

import hashlib
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = PROJECT_ROOT / "data_merged"
NEW_DATASET = PROJECT_ROOT / "public_datasets" / "book_models_v5"
OUTPUT_DATASET = PROJECT_ROOT / "data_merged_v3"

SOURCE_BOOK_CLASS = 2
TARGET_BOOK_CLASS = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_hash(image_path: Path) -> str:
    """计算图片哈希，用于排除重复图片。"""
    digest = hashlib.sha256()
    with image_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_book_label(label_path: Path) -> list[str]:
    """只保留源类别 Book，并映射为项目类别 book=1。"""
    converted = []

    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        parts = raw_line.split()
        if not parts:
            continue
        try:
            source_class = int(float(parts[0]))
        except ValueError as error:
            raise ValueError(f"类别编号错误：{label_path}:{line_number}") from error

        if source_class != SOURCE_BOOK_CLASS:
            continue

        coordinates = [float(value) for value in parts[1:]]
        if len(coordinates) == 4:
            # 检测框格式：中心 x、中心 y、宽度和高度。
            parts[0] = str(TARGET_BOOK_CLASS)
            converted.append(" ".join(parts))
            continue

        if len(coordinates) < 6 or len(coordinates) % 2 != 0:
            raise ValueError(f"标签格式错误：{label_path}:{line_number}")

        # 多边形格式：将所有顶点转换为外接矩形检测框。
        x_values = coordinates[0::2]
        y_values = coordinates[1::2]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        box = [
            TARGET_BOOK_CLASS,
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            x_max - x_min,
            y_max - y_min,
        ]
        converted.append(" ".join(f"{value:.8f}" for value in box))

    return converted


def prepare_output() -> None:
    """复制旧数据集，并加入新的 book 图片。"""
    if OUTPUT_DATASET.exists():
        raise FileExistsError(f"输出目录已存在，请先人工检查：{OUTPUT_DATASET}")
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"旧数据集不存在：{SOURCE_DATASET}")
    if not NEW_DATASET.exists():
        raise FileNotFoundError(f"新数据集不存在：{NEW_DATASET}")

    image_hashes = set()

    for split in ("train", "val", "test"):
        source_images = SOURCE_DATASET / "images" / split
        source_labels = SOURCE_DATASET / "labels" / split
        target_images = OUTPUT_DATASET / "images" / split
        target_labels = OUTPUT_DATASET / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(source_images.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"旧数据集缺少标签：{label_path}")

            shutil.copy2(image_path, target_images / image_path.name)
            shutil.copy2(label_path, target_labels / label_path.name)
            image_hashes.add(image_hash(image_path))

    added = 0
    skipped_without_book = 0
    skipped_duplicate = 0

    for source_split, target_split in (
        ("train", "train"),
        ("valid", "val"),
        ("test", "test"),
    ):
        source_images = NEW_DATASET / source_split / "images"
        source_labels = NEW_DATASET / source_split / "labels"
        target_images = OUTPUT_DATASET / "images" / target_split
        target_labels = OUTPUT_DATASET / "labels" / target_split

        for image_path in sorted(source_images.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"新数据集缺少标签：{label_path}")

            converted = convert_book_label(label_path)
            if not converted:
                skipped_without_book += 1
                continue

            current_hash = image_hash(image_path)
            if current_hash in image_hashes:
                skipped_duplicate += 1
                continue

            output_stem = f"book_models_v5_{image_path.stem}"
            output_image = target_images / f"{output_stem}{image_path.suffix.lower()}"
            output_label = target_labels / f"{output_stem}.txt"

            shutil.copy2(image_path, output_image)
            output_label.write_text("\n".join(converted) + "\n", encoding="utf-8")
            image_hashes.add(current_hash)
            added += 1

    write_dataset_yaml()
    write_sources(added, skipped_without_book, skipped_duplicate)
    validate_output()


def write_dataset_yaml() -> None:
    """写入新的三类别数据集配置。"""
    yaml_text = f"""path: {OUTPUT_DATASET.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: cup
  1: book
  2: pen
"""
    (OUTPUT_DATASET / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_sources(added: int, skipped_without_book: int, skipped_duplicate: int) -> None:
    """记录新数据集来源和本次处理结果。"""
    source_text = (SOURCE_DATASET / "SOURCES.md").read_text(encoding="utf-8")
    source_text += (
        "\n\n## 本次新增来源\n\n"
        "- `book_models_v5`: Public Domain, "
        "https://universe.roboflow.com/bukuaok/book-models-ww1pt/dataset/5\n"
        f"- 加入图片：{added}\n"
        f"- 跳过无 Book 标注图片：{skipped_without_book}\n"
        f"- 跳过重复图片：{skipped_duplicate}\n"
    )
    (OUTPUT_DATASET / "SOURCES.md").write_text(source_text, encoding="utf-8")


def validate_output() -> None:
    """检查输出数据集的图片和标签是否一一对应。"""
    for split in ("train", "val", "test"):
        image_dir = OUTPUT_DATASET / "images" / split
        label_dir = OUTPUT_DATASET / "labels" / split
        image_stems = {
            path.stem
            for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        }
        label_stems = {path.stem for path in label_dir.glob("*.txt")}
        if image_stems != label_stems:
            raise RuntimeError(f"{split} 图片和标签不匹配")
        print(f"{split}: {len(image_stems)} 张图片，{len(label_stems)} 个标签")


if __name__ == "__main__":
    prepare_output()
