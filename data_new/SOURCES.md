# data_new 数据来源

类别编号固定为：

- `0`: cup
- `1`: book
- `2`: pen

## 使用的数据

- 原 cup 图片：从 `data_old` 中保留仅含 cup 标注的图片。
- 新 book 数据：`book.v1i.yolov8.zip`，CC BY 4.0。
  https://universe.roboflow.com/santiago-garcia/book-urxgh/dataset/1
- 新 pen 数据：`Pen-Tracking-Coursework.v5i.yolov8.zip`，CC BY 4.0。
  https://universe.roboflow.com/coco-oshnc/pen-tracking-coursework/dataset/5

## 合并规则

- 旧数据中的 book 和 pen 图片没有加入。
- 新数据的 `book` 源类别 `0` 映射为项目类别 `1`。
- 新数据的 `Pen` 源类别 `0` 映射为项目类别 `2`。
- 图片和标签按 SHA-256 去除完全重复的图片。
- 新数据优先保留原有 train、valid、test 划分；如果压缩包没有 valid，则从 train 固定划出 20% 作为 val。

## 加入数量

| 来源 | train | val | test |
|---|---:|---:|---:|
| 原 cup | 186 | 13 | 7 |
| 新 book | 765 | 191 | 106 |
| 新 pen | 2163 | 167 | 118 |
