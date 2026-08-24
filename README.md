# Desktop Detection V1

基于 Ultralytics YOLO 的桌面物体检测学习项目，用于检测图片中的：

- `cup`
- `book`
- `pen`

项目完整实现了从数据标注、YOLO 标签转换、多个数据集合并、模型训练、测试集评估到图片推理的流程。

## 项目结果

仓库名称 `Desktop_Detection_V1` 表示这是项目的第一个正式发布版本。训练过程中曾使用过 `desktop_merged_v1`、`desktop_merged_v2`、`desktop_merged_v3` 等实验名称；最终选中的权重已经统一重命名为正式发布名称：

```text
models/final/desktop_detection_v1.pt
```

在扩展后的测试集 `data_merged_v3` 上评估结果如下：

| 类别 | mAP50-95 |
| --- | ---: |
| cup | 0.8282 |
| book | 0.2861 |
| pen | 0.5316 |
| 全部类别 | 0.5486 |

总体指标：

```text
Precision: 0.796
Recall:    0.649
mAP50:     0.7088
mAP50-95:  0.5486
```

`cup` 效果较好，`pen` 为中等水平，`book` 仍然是当前主要改进方向。最终发布模型来自历史实验 `desktop_merged_v1`；加入新 Book 数据后得到的 `desktop_merged_v3` 整体指标略低，因此没有作为正式模型发布。

## 目录结构

```text
Desktop_Detection_V1/
├── models/final/              正式发布的模型权重
├── scripts/
│   ├── son_to_yolo.py         LabelMe JSON 转 YOLO TXT
│   ├── merge_datasets.py      合并多个 YOLO 数据集
│   ├── merge_new_book_dataset.py  加入新的 Book 数据集
│   ├── train_yolo.py          训练三类别模型
│   ├── evaluate_test.py       使用独立测试集评估模型
│   └── inference.py           对单张图片进行推理
├── pyproject.toml             项目依赖和 uv 配置
├── uv.lock                    依赖锁定文件
└── README.md                  项目说明
```

## 环境要求

- Windows 10/11
- Python 3.11 或更高版本
- NVIDIA GPU 可选
- 推荐使用 `uv` 管理环境

安装依赖：

```powershell
uv sync
```

如果使用 NVIDIA GPU，确认 PyTorch 能识别显卡：

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 数据集

由于数据集体积较大，且部分数据来自第三方，仓库不上传原始图片和标签。请按照下面的目录准备本地数据：

```text
data/
data_merged/
data_merged_v3/
public_datasets/
```

项目使用过的数据来源：

- [Cup Auto Dataset](https://universe.roboflow.com/nicolai-hoirup-nielsen/cup-auto-dataset/dataset/1)
- [Pen Book Dataset](https://universe.roboflow.com/cursoyolo/pen-book-vg0dv/dataset/2)
- [Book Detection Dataset](https://universe.roboflow.com/peter-kafel-ddgdb/book-detection-e1luo/dataset/1)
- [Book Models v5](https://universe.roboflow.com/bukuaok/book-models-ww1pt/dataset/5)

最终类别编号必须保持：

```text
0: cup
1: book
2: pen
```

## 运行图片推理

将待检测图片放到：

```text
data/images/img030.JPG
```

运行：

```powershell
uv run python scripts/inference.py
```

检测结果保存到：

```text
outputs/image_test/
```

## 训练模型

准备好 `data_merged_v3/data.yaml` 后运行：

```powershell
uv run python scripts/train_yolo.py
```

训练脚本会根据当前环境自动选择 CUDA 或 CPU，并把训练结果保存到 `runs/train/`。

## 测试集评估

准备好测试集后运行：

```powershell
uv run python scripts/evaluate_test.py
```

评估结果会保存到 `runs/test/`。测试集评估比只看训练过程中的验证集更适合用来选择最终模型。

## 学习重点

这个项目适合按以下顺序学习：

1. Python 路径和文件操作
2. LabelMe 标注和 YOLO 标签格式
3. 数据集类别映射与训练集划分
4. CNN、YOLO 多尺度目标检测
5. Precision、Recall、mAP50、mAP50-95
6. 迁移学习和模型对比实验
7. GPU、CUDA 和 PyTorch 环境配置

## 项目限制

当前项目支持单张图片离线检测，未实现摄像头输入。没有摄像头不影响项目的核心目标检测流程；后续也可以使用视频文件扩展为离线视频检测。

## 许可证

本项目代码使用 MIT License。第三方数据集的许可证以各数据集发布页面为准。
