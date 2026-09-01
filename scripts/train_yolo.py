"""训练三类别桌面物体检测模型。"""

from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "base" / "yolov8n.pt"

DATA_PATH = PROJECT_ROOT / "data_new" / "data.yaml"
PROJECT_PATH = PROJECT_ROOT / "runs" / "train"


def train() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型不存在：{MODEL_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据集配置不存在：{DATA_PATH}")

    model = YOLO(str(MODEL_PATH))

    model.train(
        data=str(DATA_PATH),
        epochs=100,
        patience=20,
        imgsz=640,
        batch=4,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(PROJECT_PATH),
        name="desktop_detection_v2",
        exist_ok=False,
        workers=0,
        seed=42,
    )


if __name__ == "__main__":
    train()
