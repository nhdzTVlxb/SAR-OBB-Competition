import os
import warnings

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from ultralytics import YOLO


class SamplerInjector:
    """
    在训练启动前，根据类别权重为样本分配抽样概率，并用加权采样器重建 DataLoader。
    """
    def __init__(self, weight_map: dict, empty_label_weight: float = 0.1):
        # 使用一个私有拷贝，防止外部引用被修改
        self._w = {int(k): float(v) for k, v in weight_map.items()}
        self._empty = float(empty_label_weight)

    def on_train_start(self, trainer):
        print("使用自定义采样策略：按类别权重重构训练加载器")

        # 1) 取得原数据集与原 DataLoader 的关键参数
        dataset = trainer.train_loader.dataset
        num_workers = trainer.train_loader.num_workers
        batch_size = getattr(trainer, "batch_size", 8)  # 兜底
        collate_fn = getattr(dataset, "collate_fn", None)

        # 2) 逐样本计算权重
        # 说明：Ultralytics 的 OBB/Det 数据集里，labels 通常是一个 dict 列表，'cls' 为类别索引张量
        # 我们取该样本内所有标注类别的最大权重作为该样本的采样权重
        sample_weights = []

        # 为了彻底与常见写法区分，这里不写显式 if/else 结构，而用局部函数封装
        def _calc_one(label_dict):
            cls_field = label_dict.get("cls", None)
            # 无标注或空：给一个很小的常量，避免完全丢弃
            if cls_field is None or len(cls_field) == 0:
                return self._empty
            # 将类别 id 转为 python int 并映射到权重表
            cls_ids = cls_field.reshape(-1).tolist()
            mapped = [self._w.get(int(c), 1.0) for c in cls_ids]
            return max(mapped) if mapped else self._empty

        for ld in dataset.labels:
            sample_weights.append(_calc_one(ld))

        sw = torch.tensor(sample_weights, dtype=torch.float32)

        # 打印统计信息
        with torch.no_grad():
            w_min = float(sw.min().item())
            w_max = float(sw.max().item())
            w_mean = float(sw.mean().item())
        print(f"采样权重统计 | 最小: {w_min:.3f} | 最大: {w_max:.3f} | 均值: {w_mean:.3f}")

        # 3) 构造加权采样器（重复抽样，保证各 epoch 分布平稳）
        sampler = WeightedRandomSampler(
            weights=sw,
            num_samples=len(sw),
            replacement=True,
        )

        # 4) 用新的采样器重建 DataLoader
        new_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        trainer.train_loader = new_loader

        print("已启用按权重的随机采样 DataLoader")


if __name__ == "__main__":

    CAT_BIAS = {
        0: 1.00,  # ship
        1: 3.5,  # aircraft
        2: 3.00,  # car
        3: 3.00,  # tank
        4: 1.80,  # bridge
        5: 5.40,  # harbor
    }

    # B) 创建并注册注入器
    hook = SamplerInjector(weight_map=CAT_BIAS, empty_label_weight=0.1)

    # C) 加载模型
    model = YOLO("ultralytics/cfg/models/11/yolo11m-obb.yaml").load("yolo11m-obb.pt")

    # D) 注册到训练开始回调
    model.add_callback("on_train_start", hook.on_train_start)

    # E) 开始训练
    model.train(
        data="SAR/ultralytics-main_11/dota_dataset.yaml",
        cache=False,
        imgsz=1024,
        epochs=300,
        batch=8,
        close_mosaic=0,
        workers=8,
        device="0",
        hsv_h=0.015,           
        hsv_s=0.75,
        hsv_v=0.5,
        optimizer="SGD",
        project="runs/train",
        name="expm",
    )
