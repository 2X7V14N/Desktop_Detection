import json
import os
from pathlib import Path

# 类别名称与编号的映射。
CLASSES = {
    'cup': 0,
    'book': 1,
    'pen': 2
}

def convert_labelme_to_yolo(json_file, output_dir):
    """将labelme的JSON文件转换成YOLO格式的txt文件(5列)"""

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图片尺寸。
    img_width = data['imageWidth']
    img_height = data['imageHeight']

    # 准备 YOLO 格式的标注。
    yolo_lines = []

    for shape in data['shapes']:
        label = shape['label']
        if label not in CLASSES:
            print(f"警告: 未知类别 '{label}' 在文件 {json_file}")
            continue

        class_id = CLASSES[label]
        points = shape['points']

        # LabelMe 的矩形框由两个点组成：[左上，右下]。
        x1, y1 = points[0]
        x2, y2 = points[1]

        # 转换成 YOLO 格式：归一化后的中心 x、中心 y、宽度和高度。
        center_x = (x1 + x2) / 2 / img_width
        center_y = (y1 + y2) / 2 / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height

        # YOLO 格式：class_id center_x center_y width height，共 5 个数字。
        yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    # 保存为 txt 文件。
    txt_filename = Path(json_file).stem + '.txt'
    txt_path = os.path.join(output_dir, txt_filename)

    with open(txt_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    return len(yolo_lines)

def main():
    # 配置路径。
    images_dir = r"D:\JetBrains\PycharmProjects\Desktop_Detection_V1\data\images"
    labels_dir = r"D:\JetBrains\PycharmProjects\Desktop_Detection_V1\data\labels"

    # 创建 labels 文件夹（如果不存在）。
    os.makedirs(labels_dir, exist_ok=True)

    # 遍历所有 JSON 文件。
    json_files = list(Path(images_dir).glob('*.json'))

    if len(json_files) == 0:
        print("错误: 在images文件夹里没找到.json文件")
        print(f"检查路径: {images_dir}")
        return

    print(f"找到 {len(json_files)} 个JSON文件,开始转换...\n")

    total_objects = 0
    for json_file in sorted(json_files):
        num_objects = convert_labelme_to_yolo(str(json_file), labels_dir)
        total_objects += num_objects
        print(f"✓ {json_file.name} → {json_file.stem}.txt ({num_objects} 个标注)")

    print(f"\n转换完成!")
    print(f"文件数量: {len(json_files)}")
    print(f"标注总数: {total_objects}")
    print(f"YOLO标注保存在: {labels_dir}")

    # 验证一个文件。
    print("\n验证样例:")
    sample_txt = os.path.join(labels_dir, "img001.txt")
    if os.path.exists(sample_txt):
        with open(sample_txt, 'r') as f:
            content = f.read()
        print(f"{sample_txt}:")
        print(content)

if __name__ == '__main__':
    main()
