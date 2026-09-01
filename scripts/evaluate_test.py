"""使用各自对应的测试集评估 V1 和 V2 模型。"""

from pathlib import Path

import torch
from ultralytics import YOLO


# 获取项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 每个模型使用自己对应的数据集，避免混淆不同数据分布的评估结果。
MODELS = {
    "desktop_detection_v1": (
        PROJECT_ROOT / "models" / "final" / "desktop_detection_v1.pt",
        PROJECT_ROOT / "data_old" / "data.yaml",
    ),
    "desktop_detection_v2": (
        PROJECT_ROOT / "models" / "final" / "desktop_detection_v2.pt",
        PROJECT_ROOT / "data_new" / "data.yaml",
    ),
}

DEVICE = 0 if torch.cuda.is_available() else "cpu"


def evaluate_model(
    model_name: str, model_path: Path, data_path: Path
) -> None:
    """评估一个模型并打印主要指标。"""
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在：{model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置不存在：{data_path}")

    print(f"\n开始评估：{model_name}")
    print(f"模型路径：{model_path}")
    print(f"数据集配置：{data_path}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
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
    for model_name, (model_path, data_path) in MODELS.items():
        evaluate_model(model_name, model_path, data_path)


if __name__ == "__main__":
    main()
