"""使用独立测试集评估最终三类别模型。"""

from pathlib import Path

import torch
from ultralytics import YOLO


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 测试集配置文件
DATA_PATH = PROJECT_ROOT / "data_merged_v3" / "data.yaml"

# 要评估的最终模型。
MODELS = {
    "desktop_detection_v1": PROJECT_ROOT / "models" / "final" / "desktop_detection_v1.pt",
}
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def evaluate_model(model_name: str, model_path: Path) -> None:
    """评估一个模型。"""
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在：{model_path}")

    print(f"\n开始评估：{model_name}")
    print(f"模型路径：{model_path}")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(DATA_PATH),
        split="test",
        imgsz=640,
        batch=4,
        device=DEVICE,
        workers=0,
        plots=True,
        project=str(PROJECT_ROOT / "runs" / "test"),
        name=model_name,
        exist_ok=True,
    )

    print(f"\n{model_name} 评估完成")
    print(f"mAP50：{metrics.box.map50:.4f}")
    print(f"mAP50-95：{metrics.box.map:.4f}")

    print("每个类别的 mAP50-95：")
    for class_id, class_name in model.names.items():
        print(f"{class_id}: {class_name} = {metrics.box.maps[class_id]:.4f}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据集配置不存在：{DATA_PATH}")

    for model_name, model_path in MODELS.items():
        evaluate_model(model_name, model_path)


if __name__ == "__main__":
    main()
