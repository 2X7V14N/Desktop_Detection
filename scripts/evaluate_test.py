"""使用独立测试集评估 Desktop Detection V3。"""

from pathlib import Path

import torch
from ultralytics import YOLO


# 获取项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final" / "desktop_detection_v3.pt"
DATA_PATH = PROJECT_ROOT / "data_new" / "data.yaml"
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main() -> None:
    """在 data_new 的 test 集上评估 V3。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型不存在：{MODEL_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据集配置不存在：{DATA_PATH}")

    print("开始评估：desktop_detection_v3")
    print(f"模型路径：{MODEL_PATH}")
    print(f"数据集配置：{DATA_PATH}")

    model = YOLO(str(MODEL_PATH))
    metrics = model.val(
        data=str(DATA_PATH),
        split="test",
        imgsz=640,
        batch=4,
        device=DEVICE,
        workers=0,
        plots=True,
        project=str(PROJECT_ROOT / "runs" / "test"),
        name="desktop_detection_v3",
        exist_ok=True,
    )

    print("\nDesktop Detection V3 评估完成")
    print(f"Precision：{metrics.box.mp:.4f}")
    print(f"Recall：{metrics.box.mr:.4f}")
    print(f"mAP50：{metrics.box.map50:.4f}")
    print(f"mAP50-95：{metrics.box.map:.4f}")

    print("每个类别的 mAP50-95：")
    for class_id, class_name in model.names.items():
        print(f"{class_id}: {class_name} = {metrics.box.maps[class_id]:.4f}")


if __name__ == "__main__":
    main()
