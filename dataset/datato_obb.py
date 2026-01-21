import os
import glob


def convert_dota_to_yolo_obb(input_folder, output_folder=None):
    """
    将 DOTA 格式标注转换为 YOLO-OBB：
    输出行格式为：class_index x1 y1 x2 y2 x3 y3 x4 y4
    """
    class_mapping = {
        'ship': 0,
        'aircraft': 1,
        'car': 2,
        'tank': 3,
        'bridge': 4,
        'harbor': 5
    }

    if output_folder is None:
        output_folder = input_folder
    else:
        os.makedirs(output_folder, exist_ok=True)

    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
    if not txt_files:
        print(f"在文件夹 {input_folder} 中没有找到 txt 文件")
        return

    converted_count = 0
    error_count = 0

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            converted_lines = []

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 10:
                    print(f"警告: {txt_file} 第{line_num}行数据格式不正确: {line}")
                    continue

                x1, y1, x2, y2, x3, y3, x4, y4 = parts[0:8]
                classname = parts[8]

                if classname not in class_mapping:
                    print(f"警告: {txt_file} 第{line_num}行包含未知类别: {classname}")
                    continue

                class_index = class_mapping[classname]
                new_line = f"{class_index} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"
                converted_lines.append(new_line)

            output_file = os.path.join(output_folder, os.path.basename(txt_file))
            with open(output_file, 'w', encoding='utf-8') as f:
                for l in converted_lines:
                    f.write(l + '\n')

            converted_count += 1
            print(f"已转换: {txt_file} -> {output_file} ({len(converted_lines)} 个目标)")

        except Exception as e:
            error_count += 1
            print(f"错误: 无法处理文件 {txt_file}: {e}")

    print("\n转换完成!")
    print(f"成功转换: {converted_count} 个文件")
    print(f"转换失败: {error_count} 个文件")
    print(f"类别映射: {class_mapping}")


def main():
    input_folder = ''
    output_folder = ''
    convert_dota_to_yolo_obb(input_folder, output_folder)


if __name__ == "__main__":
    main()
