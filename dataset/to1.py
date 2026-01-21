import os
import cv2
from pathlib import Path


def normalize_yolo_labels(images_folder, labels_folder, output_labels_folder=None):
    """
    将 YOLO-OBB 标签中的坐标归一化到 0-1 之间。

    Args:
        images_folder (str): 图片文件夹路径 (包含 png 文件)
        labels_folder (str): 标签文件夹路径 (包含 txt 文件)
        output_labels_folder (str): 输出标签文件夹路径；None 表示覆盖原标签
    """
    if output_labels_folder is None:
        output_labels_folder = labels_folder
    else:
        os.makedirs(output_labels_folder, exist_ok=True)

    label_files = list(Path(labels_folder).glob("*.txt"))
    if not label_files:
        print(f"错误: 在文件夹 {labels_folder} 中没有找到 txt 文件")
        return

    print(f"开始处理 {len(label_files)} 个标签文件...")

    processed_count = 0
    error_count = 0
    skipped_count = 0

    for i, txt_path in enumerate(label_files):
        if (i + 1) % 1000 == 0 or i == 0:
            progress = (i + 1) / len(label_files) * 100
            print(f"进度: {i + 1}/{len(label_files)} ({progress:.1f}%)")

        try:
            image_name = txt_path.stem + ".png"
            image_path = os.path.join(images_folder, image_name)

            if not os.path.exists(image_path):
                print(f"警告: 图片 {image_path} 不存在，跳过标注文件 {txt_path.name}")
                skipped_count += 1
                continue

            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图片 {image_path}，跳过标注文件 {txt_path.name}")
                skipped_count += 1
                continue

            img_height, img_width = img.shape[:2]
            normalized_lines = []

            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 9:
                    print(f"警告: {txt_path.name} 第{line_num}行格式错误，应为9个值: {line}")
                    continue

                try:
                    class_index = int(parts[0])
                    coords = list(map(float, parts[1:9]))

                    normalized_coords = []
                    for j in range(0, 8, 2):
                        x_norm = max(0.0, min(1.0, coords[j] / img_width))
                        y_norm = max(0.0, min(1.0, coords[j + 1] / img_height))
                        normalized_coords.extend([x_norm, y_norm])

                    normalized_line = f"{class_index} " + " ".join(f"{c:.6f}" for c in normalized_coords)
                    normalized_lines.append(normalized_line)

                except ValueError as e:
                    print(f"警告: {txt_path.name} 第{line_num}行数据转换错误: {e}")
                    continue

            output_path = os.path.join(output_labels_folder, txt_path.name)
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in normalized_lines:
                    f.write(line + '\n')

            processed_count += 1

        except Exception as e:
            error_count += 1
            print(f"错误: 处理文件 {txt_path.name} 时出错: {e}")

    print(f"\n归一化处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"跳过文件: {skipped_count} 个文件 (对应图片不存在)")
    print(f"处理失败: {error_count} 个文件")

    if output_labels_folder != labels_folder:
        print(f"输出目录: {output_labels_folder}")
    else:
        print("已覆盖原标签文件")


def normalize_dataset_folders(dataset_folder):
    """
    处理划分后的数据集文件夹 (包含 train/val 子文件夹)。

    Args:
        dataset_folder (str): 数据集根目录，包含 images/ 和 labels/ 子文件夹
    """
    for subset in ['train', 'val']:
        images_path = os.path.join(dataset_folder, 'images', subset)
        labels_path = os.path.join(dataset_folder, 'labels', subset)

        if os.path.exists(images_path) and os.path.exists(labels_path):
            print(f"\n处理 {subset} 数据集...")
            normalize_yolo_labels(images_path, labels_path)
        else:
            print(f"跳过 {subset}: 文件夹不存在")


def main():
    print("YOLO-OBB 坐标归一化工具")
    print("=" * 50)

    images_folder = r"SAR\train\train001\images"
    labels_folder = r"SAR\train\train.zip\train\annfiles"
    output_labels_folder = None

    normalize_yolo_labels(images_folder, labels_folder, output_labels_folder)


if __name__ == "__main__":
    main()
