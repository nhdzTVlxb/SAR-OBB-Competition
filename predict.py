import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import numpy as np
import pickle
from pathlib import Path
import os
import glob

def check_model_classes(model_path):
    """检查模型类别信息"""
    model = YOLO(model_path)
    print("模型类别信息:")
    for idx, name in model.names.items():
        print(f"  索引 {idx}: {name}")
    return model.names

def convert_yolo_to_pkl(model_path, test_dir, output_pkl_path, conf_threshold=0.001):
    """
    将YOLO推理结果转换为比赛要求的pkl格式
    """
    model = YOLO(model_path)

    # 比赛类别映射
    class_mapping = {
        0: 'ship',
        1: 'aircraft',
        2: 'car',
        3: 'tank',
        4: 'bridge',
        5: 'harbor'
    }

    # 获取所有测试图片
    test_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        pattern = os.path.join(test_dir, ext)
        test_images.extend(glob.glob(pattern))

    # 按文件名数字排序
    test_images = sorted(test_images, key=lambda x: int(os.path.basename(x).split('.')[0]))
    print(f"找到 {len(test_images)} 张测试图片")

    all_results = []

    # 推理每张图片
    for i, img_path in enumerate(test_images):
        img_name = os.path.basename(img_path)
        print(f"处理图片 ({i+1}/{len(test_images)}): {img_name}")

        # 初始化结果字典
        image_result = {
            'image': img_name,
            'poly': np.empty((0, 8), dtype=float),
            'scores': [],
            'labels': []
        }

        try:
            results = model(img_path, conf=conf_threshold, iou=0.75, max_det=600, verbose=False)
            result = results[0]

            if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
                confs = result.obb.conf.cpu().numpy()
                classes = result.obb.cls.cpu().numpy()

                polygons, scores, labels = [], [], []
                for j in range(len(result.obb)):
                    poly_points = xyxyxyxy[j].flatten()
                    conf_score = float(confs[j])
                    class_idx = int(classes[j])

                    if class_idx in class_mapping:
                        class_name = class_mapping[class_idx]
                        polygons.append(poly_points)
                        scores.append(conf_score)
                        labels.append(class_name)

                if len(polygons) > 0:
                    image_result['poly'] = np.array(polygons, dtype=float)
                    image_result['scores'] = scores
                    image_result['labels'] = labels

        except Exception as e:
            print(f"  处理图片 {img_name} 时出错: {e}")

        all_results.append(image_result)

        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1} 张图片，保存临时进度...")
            with open(output_pkl_path + ".temp", "wb") as f:
                pickle.dump(all_results, f)

    with open(output_pkl_path, "wb") as f:
        pickle.dump(all_results, f)

    print(f"\n结果已保存到: {output_pkl_path}")
    return all_results

def verify_pkl_format(pkl_path):
    """验证pkl文件格式"""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        print(f"\n验证pkl文件: {pkl_path}")
        print(f"总图片数: {len(data)}")
        file_size = os.path.getsize(pkl_path) / 1024 / 1024
        print(f"文件大小: {file_size:.2f} MB")

        total_detections = sum(len(entry['scores']) for entry in data)
        print(f"总检测目标数: {total_detections}")

        for i, entry in enumerate(data[:5]):
            print(f"\n图片 {i+1}: {entry['image']}")
            print(f"  检测数量: {len(entry['scores'])}")
            if len(entry['scores']) > 0:
                print(f"  置信度样例: {entry['scores'][:3]}")
                print(f"  类别样例: {entry['labels'][:3]}")

    except Exception as e:
        print(f"验证pkl文件时出错: {e}")

if __name__ == "__main__":
    # 路径配置
    model_path = r"C:\Users\Mayn\Desktop\ultralytics-main\runs\train\expm\weights\best.pt"
    test_dir = r"D:\AAA_SAR\test_B_images\images"#
    output_pkl = "best.pkl"

    if not os.path.exists(test_dir):
        print(f"错误: 测试目录不存在: {test_dir}")
        exit()

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        exit()

    print("检查模型类别:")
    _ = check_model_classes(model_path)

    results = convert_yolo_to_pkl(model_path, test_dir, output_pkl, conf_threshold=0.001)

    verify_pkl_format(output_pkl)
