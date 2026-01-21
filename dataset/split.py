import os
import shutil
import random
from pathlib import Path

def split_dataset_yolo_obb(source_images_dir, source_labels_dir, output_dir, train_ratio=0.8, seed=42):
    """
    将YOLO-OBB数据集按8:2分割为train/val
    
    Args:
        source_images_dir (str): 源图片文件夹路径
        source_labels_dir (str): 源标签文件夹路径  
        output_dir (str): 输出根目录
        train_ratio (float): 训练集比例，默认0.8
        seed (int): 随机种子
    """
    
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录结构
    images_train_dir = os.path.join(output_dir, 'images', 'train')
    images_val_dir = os.path.join(output_dir, 'images', 'val')
    labels_train_dir = os.path.join(output_dir, 'labels', 'train')
    labels_val_dir = os.path.join(output_dir, 'labels', 'val')
    
    for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 获取所有图片文件名（不含扩展名）
    image_files = []
    for file in os.listdir(source_images_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_files.append(Path(file).stem)
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    if not image_files:
        print("错误: 没有找到图片文件")
        return
    
    # 随机打乱并分割
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    
    # 复制文件函数
    def copy_files(file_list, images_dest, labels_dest):
        copied_count = 0
        for file_stem in file_list:
            try:
                # 查找图片文件（支持多种格式）
                image_found = False
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    source_image = os.path.join(source_images_dir, file_stem + ext)
                    if os.path.exists(source_image):
                        shutil.copy2(source_image, os.path.join(images_dest, file_stem + ext))
                        image_found = True
                        break
                
                if not image_found:
                    print(f"警告: 未找到图片文件 {file_stem}")
                    continue
                
                # 复制标签文件
                source_label = os.path.join(source_labels_dir, file_stem + '.txt')
                if os.path.exists(source_label):
                    shutil.copy2(source_label, os.path.join(labels_dest, file_stem + '.txt'))
                else:
                    print(f"警告: 未找到标签文件 {file_stem}.txt")
                    continue
                
                copied_count += 1
                
            except Exception as e:
                print(f"错误: 复制文件 {file_stem} 时出错: {e}")
        
        return copied_count
    
    # 复制训练集文件
    print("正在复制训练集文件...")
    train_copied = copy_files(train_files, images_train_dir, labels_train_dir)
    
    # 复制验证集文件
    print("正在复制验证集文件...")
    val_copied = copy_files(val_files, images_val_dir, labels_val_dir)
    
    print(f"\n分割完成!")
    print(f"训练集: {train_copied} 个文件")
    print(f"验证集: {val_copied} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"目录结构:")
    print(f"  {output_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/    # {len(os.listdir(images_train_dir))} 个图片文件")
    print(f"  │   └── val/      # {len(os.listdir(images_val_dir))} 个图片文件")
    print(f"  └── labels/")
    print(f"      ├── train/    # {len(os.listdir(labels_train_dir))} 个标签文件")
    print(f"      └── val/      # {len(os.listdir(labels_val_dir))} 个标签文件")

def main():
    """
    主函数 - 配置路径并执行分割
    """
    # 配置路径 - 根据你的实际情况修改
    source_images_dir = "SAR/split/images"      # 源图片文件夹
    source_labels_dir = "SAR/split/labels"      # 源标签文件夹
    output_dir = "SAR/split"                    # 输出根目录
    
    # 检查源目录是否存在
    if not os.path.exists(source_images_dir):
        print(f"错误: 图片目录不存在 {source_images_dir}")
        return
    
    if not os.path.exists(source_labels_dir):
        print(f"错误: 标签目录不存在 {source_labels_dir}")
        return
    
    print("YOLO-OBB数据集8:2分割工具")
    print("=" * 50)
    print(f"源图片目录: {source_images_dir}")
    print(f"源标签目录: {source_labels_dir}")
    print(f"输出目录: {output_dir}")
    print(f"分割比例: 80% 训练集, 20% 验证集")
    print("=" * 50)
    
    # 执行分割
    split_dataset_yolo_obb(source_images_dir, source_labels_dir, output_dir)

if __name__ == "__main__":
    main()