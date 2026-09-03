# data_new 数据来源

## 项目类别

- `0`: cup
- `1`: book
- `2`: pen

## 数据来源

- 原 cup 图片：从旧合并数据中保留的 cup 数据。
- 新 book 数据：`book.v1i.yolov8.zip`，CC BY 4.0。
  https://universe.roboflow.com/santiago-garcia/book-urxgh/dataset/1
- 新 pen 数据：`pen.v1-pen-version2.yolov8.zip`，CC BY 4.0。
  https://universe.roboflow.com/yolov5-h3oo1/pen-zyj9j/dataset/1

## 合并规则

- book 和 cup 数据保持原样，没有修改或重新划分。
- 原 data_new 中的 pen 数据全部移除。
- 新数据集的 `Pen` 源类别 `0` 映射为项目类别 `2`。
- 新 pen 数据原本只有 `train` 和 `valid`，按固定 SHA-256 排序后重新划分为 train/val/test，比例约为 80%/10%/10%。
- 所有图片都复制到统一的 `images/{train,val,test}` 目录，标签复制到对应的 `labels/{train,val,test}` 目录。

## 数据数量

| 类别 | train | val | test | 总计 |
|---|---:|---:|---:|---:|
| cup | 186 | 13 | 7 | 206 |
| book | 765 | 191 | 106 | 1062 |
| 新 pen | 668 | 83 | 85 | 836 |
| 合计图片 | 1619 | 287 | 198 | 2104 |

新 pen 数据集的原始导出中包含 806 张 train 图片和 30 张 valid 图片，共 836 张；本项目将其统一整理为三部分，便于训练、验证和最终测试。
