
import os, json, random, traceback
from datetime import datetime
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

from ultralytics import YOLO


DATA_YAML   = "/home/ultralytics-main_11/dota_datasetx.yaml"
START_BEST  = "/home/ultralytics-main_11/runs/train/expm5/weights/best.pt"
PROJECT_DIR = "runs/refine_v6_final"

DEVICE      = "1"   
WORKERS     = 8

# 验证口径
VAL_IMGSZ   = 1024
VAL_CONF    = 0.001
VAL_IOU     = 0.65
VAL_MAXDET  = 600

# 接受新权重门槛
DELTA_MAP_OK  = 0.001
DELTA_WEAK_AP = 0.5
WEAK_CLASS_IDS = {4, 5}  # 4=bridge, 5=harbor

# 阶段开关
ENABLE_B1_BALANCED = True
ENABLE_E1_HARDMIX  = True

# 批量（OOM 自动降批）
BATCH_D1 = 6   # imgsz=1216
BATCH_D2 = 12 # imgsz=1024

# ================ Utils ================
def now_tag():
    return datetime.now().strftime("%m%d_%H%M")

def read_yaml(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if yaml is not None:
        try:
            return yaml.safe_load(text)
        except Exception:
            pass
    # 简易回退（不解析 names 结构）
    data = {}
    for k in ("train","val","test","nc","names"):
        for line in text.splitlines():
            if line.strip().startswith(f"{k}:"):
                data[k] = line.split(":",1)[1].strip()
                break
    return data

def load_class_names():
    names = ["ship","aircraft","car","tank","bridge","harbor"]  # 默认兜底
    try:
        d = read_yaml(DATA_YAML)
        n = d.get("names", None)
        if isinstance(n, dict):
            names = [n[k] for k in sorted(n.keys(), key=lambda x: int(x))]
        elif isinstance(n, list):
            names = list(n)
    except Exception:
        pass
    return names

CLASS_NAMES = load_class_names()

def safe_train(yolo: YOLO, name: str, **kwargs):
    """带 OOM 处理的 train，若 batch 太大自动减半重试（>=2）。"""
    batch = int(kwargs.get("batch", 4))
    while batch >= 2:
        try:
            print(f"\n[TRAIN] {name} with batch={batch}")
            kwargs["batch"] = batch
            yolo.train(name=name, project=PROJECT_DIR, exist_ok=True, workers=WORKERS, **kwargs)
            save_dir = Path(PROJECT_DIR) / name
            best_path = save_dir / "weights" / "best.pt"
            if not best_path.exists():
                raise FileNotFoundError(f"未找到 {best_path}")
            return str(best_path)
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg and batch > 2:
                batch = max(2, batch // 2)
                print(f"[WARN] OOM, 降低 batch 到 {batch} 重试…")
                continue
            print("[ERROR] 训练失败：", msg)
            raise
    raise RuntimeError("batch 太小仍 OOM，请手动调小 imgsz 或释放显存")

def extract_metrics(res):
    """稳健的指标提取：优先 res.metrics.box，兼容 8.3.199/202；按类 AP 若可得则返回。"""
    # 优先从 res.metrics.box 取（官方对象）
    m = getattr(res, "metrics", None)
    if m and hasattr(m, "box"):
        box = m.box
        maps = getattr(box, "maps", []) or []
        return {
            "map": float(getattr(box, "map", 0.0) or 0.0),
            "map50": float(getattr(box, "map50", 0.0) or 0.0),
            "mp": float(getattr(box, "mp", 0.0) or 0.0),
            "mr": float(getattr(box, "mr", 0.0) or 0.0),
            "maps": [float(x) for x in maps],
        }
    # 其次从 results_dict 兜底
    d = getattr(res, "results_dict", {}) or {}
    return {
        "map": float(d.get("mAP50-95", d.get("mAP50-95(B)", 0.0)) or 0.0),
        "map50": float(d.get("mAP50", d.get("mAP50(B)", 0.0)) or 0.0),
        "mp": float(d.get("P", 0.0) or 0.0),
        "mr": float(d.get("R", 0.0) or 0.0),
        "maps": None,
    }

def pretty_print_metrics(tag, met):
    print(f"[VAL] {tag} -> mAP50-95={met['map']:.6f}  mAP50={met['map50']:.6f}  P={met['mp']:.6f}  R={met['mr']:.6f}")
    if met.get("maps") and len(met["maps"]) == len(CLASS_NAMES):
        per = {CLASS_NAMES[i]: round(met["maps"][i], 6) for i in range(len(CLASS_NAMES))}
        if "bridge" in per or "harbor" in per:
            print("      bridge(AP50-95)=", per.get("bridge","-"), "  harbor(AP50-95)=", per.get("harbor","-"))
        else:
            print("      per-class AP:", json.dumps(per, ensure_ascii=False))

def val_model(weights, desc=""):
    m = YOLO(weights)
    res = m.val(
        data=DATA_YAML, imgsz=VAL_IMGSZ,
        conf=VAL_CONF, iou=VAL_IOU, max_det=VAL_MAXDET,
        device=DEVICE, verbose=True, plots=False
    )
    met = extract_metrics(res)
    pretty_print_metrics(desc, met)
    return met

def better(new_met, old_met):
    """是否接受新权重：整体涨 +0.001；或弱类 AP 各自 +0.5（若可得）"""
    try:
        if new_met["map"] >= (old_met["map"] + DELTA_MAP_OK):
            return True
        if new_met.get("maps") and old_met.get("maps"):
            weak_new = [new_met["maps"][i] for i in WEAK_CLASS_IDS if i < len(new_met["maps"])]
            weak_old = [old_met["maps"][i] for i in WEAK_CLASS_IDS if i < len(old_met["maps"])]
            if weak_new and weak_old and all((n - o) >= DELTA_WEAK_AP for n, o in zip(weak_new, weak_old)):
                return True
    except Exception:
        pass
    return False

def make_balanced_train_list(txt_out="train_bh130.txt", weak_ids={4,5}):
    """对包含弱类的图像 +30% 额外采样"""
    root = Path("/home/fei/赛题/赛题1/train/split")
    lab = root/"labels/train"
    img = root/"images/train"
    all_imgs, weak_imgs = [], []
    for lb in lab.glob("*.txt"):
        try:
            lines = [x.strip() for x in lb.read_text().splitlines() if x.strip()]
        except:
            continue
        has_weak = any(int(x.split()[0]) in weak_ids for x in lines if x and x[0].isdigit())
        pimg = img/f"{lb.stem}.png"
        if pimg.exists():
            all_imgs.append(str(pimg))
            if has_weak:
                weak_imgs.append(str(pimg))
    dup = int(len(weak_imgs)*0.3)
    import random as _r
    balanced = all_imgs + _r.sample(weak_imgs, min(dup, len(weak_imgs)))
    Path(txt_out).write_text("\n".join(balanced), encoding="utf-8")
    print(f"[BALANCED] 写入 {txt_out}，总图像={len(balanced)}，弱类样本={len(weak_imgs)}，+dup={dup}")
    return str(Path(txt_out).resolve())

def derive_yaml(base_yaml, new_yaml, new_train_list):
    # 用 pyyaml 写入（如可用）
    try:
        base = read_yaml(base_yaml)
        if isinstance(base, dict):
            base["train"] = new_train_list
            if yaml is not None:
                Path(new_yaml).write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
            else:
                # 简易文本替换
                txt = Path(base_yaml).read_text(encoding="utf-8", errors="ignore")
                lines = []
                for line in txt.splitlines():
                    if line.strip().startswith("train:"):
                        lines.append(f"train: {new_train_list}")
                    else:
                        lines.append(line)
                Path(new_yaml).write_text("\\n".join(lines), encoding="utf-8")
            print(f"[YAML] 写入 {new_yaml} (train: {new_train_list})")
            return new_yaml
    except Exception as e:
        print("[WARN] derive_yaml 失败，回退文本替换：", e)
    # 最终兜底
    txt = Path(base_yaml).read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in txt.splitlines():
        if line.strip().startswith("train:"):
            lines.append(f"train: {new_train_list}")
        else:
            lines.append(line)
    Path(new_yaml).write_text("\\n".join(lines), encoding="utf-8")
    print(f"[YAML] 写入 {new_yaml} (fallback替换)")
    return new_yaml

def mine_hard_samples(weights, save_txt="train_bh_hard.txt"):
    """简单挖掘弱类难样本（出现低置信度 4/5 或完全缺失的候选）"""
    print("[HARD] 挖掘弱类难样本（可选阶段，默认关闭）")
    model = YOLO(weights)
    data_root = Path("/home/fei/赛题/赛题1/train/split/images/train")
    imgs = sorted(data_root.glob("*.png"))
    hard = []
    for i, p in enumerate(imgs, 1):
        if i % 500 == 0:
            print(f"  progress: {i}/{len(imgs)}")
        try:
            r = model(str(p), imgsz=1024, conf=0.01, iou=VAL_IOU, device=DEVICE, verbose=False)[0]
            has_low_bridge = any((int(b.cls.item())==4 and b.conf.item()<0.2) for b in r.boxes)
            has_low_harbor = any((int(b.cls.item())==5 and b.conf.item()<0.2) for b in r.boxes)
            if has_low_bridge or has_low_harbor:
                hard.append(str(p))
        except Exception as e:
            print("  [WARN] 推理失败：", p, e)
    Path(save_txt).write_text("\\n".join(hard), encoding="utf-8")
    print(f"[HARD] 挖掘到 {len(hard)} 个难样本 -> {save_txt}")
    return str(Path(save_txt).resolve())

# ================ Pipeline ================
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(DEVICE.split(","))

    print("\\n[Step 0] 基线验证（微抛光前）...")
    current_best = START_BEST
    best_met = val_model(current_best, desc="BASE@current")
    leaderboard = [("BASE@current", best_met, current_best)]

    # 1) B1_balanced 1 epoch（全类）
    if ENABLE_B1_BALANCED:
        try:
            list_txt = make_balanced_train_list("train_bh130.txt", WEAK_CLASS_IDS)
            data_bh = derive_yaml(DATA_YAML, "dota_dataset_bh130.yaml", list_txt)
            name = f"exp_B1_balanced_bh130_{now_tag()}"
            best_path = safe_train(
                YOLO(current_best), name=name,
                data=data_bh, imgsz=1024, epochs=1, device=DEVICE,
                batch=BATCH_D2, freeze=10, optimizer="SGD",
                lr0=5e-5, lrf=5e-6,
                box=8.0, dfl=1.5, cls=0.6,
                mosaic=0.0, mixup=0.0, auto_augment="none",
                degrees=0.0, translate=0.05, scale=0.30, fliplr=0.5, flipud=0.0,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0,
                rect=True, deterministic=True, seed=0, warmup_epochs=0.5,
                val=True
            )
            met = val_model(best_path, desc="B1_balanced")
            leaderboard.append(("B1_balanced", met, best_path))
            if better(met, best_met):
                current_best, best_met = best_path, met
                print("[ACCEPT] 采用 B1_balanced 作为新 best")
            else:
                print("[REJECT] B1_balanced 未超过基线，回退")
        except Exception as e:
            print("[SKIP] B1_balanced 失败：", e)

    # 2) 可选：E1_hardmix 1 epoch
    if ENABLE_E1_HARDMIX:
        try:
            hard_txt = mine_hard_samples(current_best, "train_bh_hard.txt")
            base_list = Path("train_bh130.txt").read_text().splitlines() if Path("train_bh130.txt").exists() else []
            hard = Path(hard_txt).read_text().splitlines()
            mix = base_list + hard[: max(1, int(0.5*len(hard))) ]
            Path("train_bh_hardmix.txt").write_text("\\n".join(mix), encoding="utf-8")
            data_hard = derive_yaml(DATA_YAML, "dota_dataset_bh_hardmix.yaml", "train_bh_hardmix.txt")
            name = f"exp_E1_hardmix_polish_{now_tag()}"
            best_path = safe_train(
                YOLO(current_best), name=name,
                data=data_hard, imgsz=1024, epochs=1, device=DEVICE,
                batch=BATCH_D2, freeze=10, optimizer="SGD",
                lr0=5e-5, lrf=5e-6,
                box=8.0, dfl=1.5, cls=0.6,
                mosaic=0.0, mixup=0.0, auto_augment="none",
                degrees=0.0, translate=0.05, scale=0.30, fliplr=0.5, flipud=0.0,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0,
                rect=True, deterministic=True, seed=0, warmup_epochs=0.5,
                val=True
            )
            met = val_model(best_path, desc="E1_hardmix")
            leaderboard.append(("E1_hardmix", met, best_path))
            if better(met, best_met):
                current_best, best_met = best_path, met
                print("[ACCEPT] 采用 E1_hardmix 作为新 best")
            else:
                print("[REJECT] E1_hardmix 未超过当前 best，回退")
        except Exception as e:
            print("[SKIP] E1_hardmix 失败：", e)

    # 3) D1×2（1216，只训颈头，轻微 mosaic，加重 box/dfl）
    try:
        name = f"exp_D1x2_box9_dfl17_{now_tag()}"
        best_path = safe_train(
            YOLO(current_best), name=name,
            data=DATA_YAML, imgsz=1216, epochs=2, device=DEVICE,
            batch=BATCH_D1, freeze=10, optimizer="SGD",
            lr0=5e-5, lrf=5e-6,
            box=9.0, dfl=1.7, cls=0.6,
            mosaic=0.05, close_mosaic=1, mixup=0.0, auto_augment="none",
            degrees=1.5, translate=0.05, scale=0.30, shear=0.0,
            fliplr=0.5, flipud=0.0,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.10,
            rect=True, deterministic=True, seed=0, warmup_epochs=0.5,
            val=True
        )
        met = val_model(best_path, desc="D1x2_box9_dfl17")
        leaderboard.append(("D1x2_box9_dfl17", met, best_path))
        if better(met, best_met):
            current_best, best_met = best_path, met
            print("[ACCEPT] 采用 D1x2 作为新 best")
        else:
            print("[REJECT] D1x2 未超过当前 best，回退")
    except Exception as e:
        print("[SKIP] D1x2 失败：", e)

    # 4) D2×1（1024 固化，无拼图/形变）
    try:
        name = f"exp_D2_box9_dfl17_{now_tag()}"
        best_path = safe_train(
            YOLO(current_best), name=name,
            data=DATA_YAML, imgsz=1024, epochs=1, device=DEVICE,
            batch=BATCH_D2, freeze=10, optimizer="SGD",
            lr0=5e-5, lrf=5e-6,
            box=9.0, dfl=1.7, cls=0.6,
            mosaic=0.0, mixup=0.0, auto_augment="none",
            degrees=0.0, translate=0.05, scale=0.30,
            fliplr=0.5, flipud=0.0,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0,
            rect=True, deterministic=True, seed=0, warmup_epochs=0.5,
            val=True
        )
        met = val_model(best_path, desc="D2_box9_dfl17")
        leaderboard.append(("D2_box9_dfl17", met, best_path))
        if better(met, best_met):
            current_best, best_met = best_path, met
            print("[ACCEPT] 采用 D2_box9_dfl17 作为新 best")
        else:
            print("[REJECT] D2_box9_dfl17 未超过当前 best，回退")
    except Exception as e:
        print("[SKIP] D2_box9_dfl17 失败：", e)

    # 5) Leaderboard
    print("\\n===== Leaderboard（按 map 降序 | imgsz=1024 | conf=%.3f | iou=%.2f | max_det=%d）=====" % (VAL_CONF, VAL_IOU, VAL_MAXDET))
    leaderboard_sorted = sorted(leaderboard, key=lambda x: (x[1]["map"] if x[1]["map"] is not None else -1), reverse=True)
    for rank, (name, met, path) in enumerate(leaderboard_sorted, 1):
        print(json.dumps({
            "rank": rank, "name": name,
            "map": round(met["map"] if met["map"] else 0.0, 6),
            "map50": round(met["map50"] if met["map50"] else 0.0, 6),
            "P": round(met["mp"] if met["mp"] else 0.0, 6),
            "R": round(met["mr"] if met["mr"] else 0.0, 6),
            "path": path
        }, ensure_ascii=False))

    best_name, best_metrics, best_path = leaderboard_sorted[0]
    print(f"\\n>>> 建议提交：{best_name} -> {best_path}（mAP50-95={best_metrics['map']:.6f}）")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\\n[FATAL] 运行失败：", e)
        traceback.print_exc()
