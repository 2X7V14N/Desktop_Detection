"""使用训练好的 YOLO 模型检测一张图片。"""

from pathlib import Path

import torch
from ultralytics import YOLO


# 项目根目录是 scripts 文件夹的上一级目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final" / "desktop_detection_v1.pt"
IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "img030.JPG"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

    # 加载训练产生的模型权重。
    model = YOLO(str(MODEL_PATH))

    # 执行检测，并保存画有检测框的图片。
    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.145,
        device=DEVICE,
        save=True,
        project=str(OUTPUT_DIR),
        name="image_test",
        exist_ok=True,
    )

    result = results[0]
    print(f"Input image: {IMAGE_PATH}")
    print(f"Annotated image: {OUTPUT_DIR / 'image_test'}")
    print(f"Detected objects: {len(result.boxes)}")

    for index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_name = result.names[class_id]

        print(
            f"{index}. {class_name} "
            f"confidence={confidence:.3f} "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )


if __name__ == "__main__":
    main()
