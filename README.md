# Desktop Detection V2

一个基于 Ultralytics YOLOv8n 的桌面物体检测学习项目，检测三类常见物体：

- `cup`
- `book`
- `pen`

项目完整覆盖数据集整理、YOLO 标签格式、模型训练、测试集评估和单张图片推理。

## V2 发布内容

V2 使用替换后的 `book` 和 `pen` 数据，并保留原有的 `cup` 数据。正式模型为：

```text
models/final/desktop_detection_v2.pt
```

V1 基线模型仍然保留，用于对比：

```text
models/final/desktop_detection_v1.pt
```

当前 GitHub 仓库地址仍为 `Desktop_Detection_V1`，这是远程仓库名称；项目内容和版本元数据已经更新为 `Desktop Detection V2`。

## V2 测试结果

V2 在 `data_new` 的独立测试集上得到：

| 类别 | mAP50-95 |
| --- | ---: |
| cup | 0.8336 |
| book | 0.9324 |
| pen | 0.5802 |
| 全部类别 | 0.7821 |

总体 `mAP50` 为 `0.9090`。其中 `pen` 仍是当前最需要改进的类别。

V1 基线在 `data_old` 测试集上的记录为：

| 类别 | mAP50-95 |
| --- | ---: |
| cup | 0.8282 |
| book | 0.2861 |
| pen | 0.5316 |
| 全部类别 | 0.5486 |

由于 V1 和 V2 使用不同测试集，这两组结果主要用于记录各自数据集上的表现，不能简单视为严格的同分布对比实验。

## 目录结构

```text
Desktop_Detection_V1/
├── data_old/                         旧版三类别数据集
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml
├── data_new/                         V2 三类别数据集
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml
├── models/
│   ├── base/yolov8n.pt               官方预训练基础权重
│   └── final/                        正式发布模型
├── scripts/
│   ├── train_yolo.py                 从 yolov8n 开始训练 V2
│   ├── evaluate_test.py              分别评估 V1 和 V2
│   └── inference.py                  单张图片推理
├── pyproject.toml                    依赖和项目版本配置
├── uv.lock                           锁定依赖版本
└── README.md                         项目说明
```

类别编号在两个数据集中保持一致：

```text
0: cup
1: book
2: pen
```

## 环境安装

要求 Windows 10/11、Python 3.11 或更高版本。项目使用 `uv` 管理依赖，并配置了 PyTorch CUDA 12.6 源。

```powershell
uv sync
```

检查 GPU 是否可用：

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果输出 `True` 和 NVIDIA 显卡名称，训练脚本会自动使用 GPU；否则使用 CPU。

## 训练 V2

`scripts/train_yolo.py` 以官方预训练权重 `models/base/yolov8n.pt` 为起点，在 `data_new/data.yaml` 上进行迁移学习，训练输出默认保存到：

```text
runs/train/desktop_detection_v2/
```

运行：

```powershell
uv run python scripts/train_yolo.py
```

训练完成后，如需更新正式模型，将：

```text
runs/train/desktop_detection_v2/weights/best.pt
```

复制为：

```text
models/final/desktop_detection_v2.pt
```

## 测试集评估

运行：

```powershell
uv run python scripts/evaluate_test.py
```

脚本会使用 `data_old` 测试 V1，使用 `data_new` 测试 V2，评估图表和日志保存在 `runs/test/`。训练输出和评估输出属于本地生成文件，不纳入版本控制。

## 单张图片推理

把待检测图片放入 `data_new/images/test/`，然后在 `scripts/inference.py` 的 `IMAGE_PATH` 中填写图片路径，运行：

```powershell
uv run python scripts/inference.py
```

默认使用 `models/final/desktop_detection_v2.pt`，结果保存到：

```text
outputs/image_test/
```

## 数据集来源

`data_old` 和 `data_new` 的整理过程及来源记录分别见对应目录中的 `SOURCES.md`。

主要公开数据来源：

- [Cup Auto Dataset](https://universe.roboflow.com/nicolai-hoirup-nielsen/cup-auto-dataset/dataset/1)
- [Pen Book Dataset](https://universe.roboflow.com/cursoyolo/pen-book-vg0dv/dataset/2)
- [Book Detection Dataset](https://universe.roboflow.com/peter-kafel-ddgdb/book-detection-e1luo/dataset/1)
- [Book Models v5](https://universe.roboflow.com/bukuaok/book-models-ww1pt/dataset/5)
- [New Book Dataset](https://universe.roboflow.com/santiago-garcia/book-urxgh/dataset/1)
- [New Pen Dataset](https://universe.roboflow.com/coco-oshnc/pen-tracking-coursework/dataset/5)

第三方数据集的许可证以原始发布页面为准。本项目代码使用 MIT License。

## 学习路线

建议按以下顺序研究：

1. Python 路径、文件和异常处理
2. YOLO 标签格式与类别编号
3. 数据集划分和数据分布
4. CNN、YOLO 多尺度检测和迁移学习
5. Precision、Recall、mAP50、mAP50-95
6. 固定随机种子的可复现实验
7. 数据增强、难例分析和模型对比

当前项目支持单张图片离线检测；没有摄像头时，也可以继续使用图片和视频文件完成算法验证。
