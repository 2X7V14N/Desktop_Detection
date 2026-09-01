# Merged dataset sources

Final classes:

- `0`: cup
- `1`: book
- `2`: pen

Included sources:

- Local dataset: `data/`, LabelMe converted to YOLO.
- `cup_dataset_2`: CC BY 4.0, https://universe.roboflow.com/nicolai-hoirup-nielsen/cup-auto-dataset/dataset/1
- `pen_book_dataset`: CC BY 4.0, https://universe.roboflow.com/cursoyolo/pen-book-vg0dv/dataset/2
- `book_detection`: MIT, https://universe.roboflow.com/peter-kafel-ddgdb/book-detection-e1luo/dataset/1

Excluded source:

- `cup_dataset_1`: its exported classes are anonymous `0`, `1`, and `2`, and inspection showed mixed cup parts/classes. It was excluded to avoid incorrect labels in the final three-class dataset.

The `book_detection` source contains `0`, `Author`, `Book`, and `Title`. Only its `Book` class was mapped to final class `1`; text-region classes were excluded. Polygon annotations were converted to enclosing YOLO detection boxes.


## 本次新增来源

- `book_models_v5`: Public Domain, https://universe.roboflow.com/bukuaok/book-models-ww1pt/dataset/5
- 加入图片：25
- 跳过无 Book 标注图片：175
- 跳过重复图片：1
