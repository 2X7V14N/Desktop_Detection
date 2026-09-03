# Desktop Detection V3

基于 Ultralytics YOLOv8n 的桌面物体检测项目，检测三类物体：`cup`、`book` 和 `pen`。

本版本使用新的 pen 数据集替换了旧 pen 数据，保留原有的 book 和 cup 数据，重点提升了笔的检测效果。

## V3 结果

V3 在 `data_new` 独立测试集上的结果：

| 类别 | mAP50-95 |
|---|---:|
| cup | 0.8150 |
| book | 0.9404 |
| pen | 0.8082 |
| 全部类别 | 0.8546 |

总体指标：

```text
Precision: 0.9857
Recall:    0.9489
mAP50:     0.9826
mAP50-95:  0.8546
```

当前正式模型：

```text
models/final/desktop_detection_v3.pt
```

## 项目结构

```text
Desktop_Detection_V1/
├── data_new/                         V3 三类别数据集
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   ├── data.yaml
│   └── SOURCES.md
├── models/
│   ├── base/yolov8n.pt               官方预训练基础模型
│   └── final/desktop_detection_v3.pt 正式模型
├── scripts/
│   ├── train_yolo.py                 训练 V3
│   ├── evaluate_test.py              测试集评估
│   └── inference.py                  单张图片推理
├── pyproject.toml                    项目依赖和版本
├── uv.lock                           依赖锁定文件
├── LICENSE                           项目许可证
└── README.md                         项目说明
```

数据集类别编号固定为：

```text
0: cup
1: book
2: pen
```

## 数据集

`data_new` 由原 cup 数据、新 book 数据和新 pen 数据组成：

| 类别 | train | val | test | 总计 |
|---|---:|---:|---:|---:|
| cup | 186 | 13 | 7 | 206 |
| book | 765 | 191 | 106 | 1062 |
| pen | 668 | 83 | 85 | 836 |
| 合计图片 | 1619 | 287 | 198 | 2104 |

book 和 cup 数据保持原样。新 pen 数据来自 `pen.v1-pen-version2.yolov8.zip`，原始类别 `0` 已映射为项目类别 `2`。详细来源和整理规则见 [data_new/SOURCES.md](data_new/SOURCES.md)。

## 环境安装

要求 Windows 10/11、Python 3.11 或更高版本。项目使用 `uv` 管理依赖，并配置 PyTorch CUDA 12.6。

```powershell
uv sync
```

检查 GPU：

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 训练

训练脚本使用 `models/base/yolov8n.pt` 作为预训练模型，在 `data_new/data.yaml` 上训练 V3：

```powershell
uv run python scripts/train_yolo.py
```

训练输出保存在 `runs/train/desktop_detection_v3/`。训练完成后，将新的 `weights/best.pt` 复制到：

```text
models/final/desktop_detection_v3.pt
```

## 测试集评估

```powershell
uv run python scripts/evaluate_test.py
```

评估使用 `data_new` 的 test 集，结果和图表保存在 `runs/test/`。

## 单张图片推理

修改 `scripts/inference.py` 中的 `IMAGE_PATH`，然后运行：

```powershell
uv run python scripts/inference.py
```

脚本默认加载 `models/final/desktop_detection_v3.pt`，检测结果保存到 `outputs/image_test/`。

## 学习内容

本项目可用于学习 Python 路径处理、YOLO 标签格式、数据集划分、CNN 特征提取、目标检测、多尺度预测、迁移学习、数据增强以及 Precision、Recall 和 mAP 指标。

当前项目支持单张图片离线检测。没有摄像头时，可以使用图片或视频文件继续验证模型。

## 许可证

项目代码使用 MIT License。第三方数据集的许可证和来源以原始发布页面及 [data_new/SOURCES.md](data_new/SOURCES.md) 为准。
