import os
import shutil
from pathlib import Path

from ultralytics import YOLO


def _project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parent


def ensure_dataset(val_ratio: float = 0.2) -> Path:
    """确保 data/dataset 目录存在且包含 data.yaml。若没有则自动调用 split_dataset.py 生成。"""
    root = _project_root()
    dataset_dir = root / 'data' / 'dataset'
    data_yaml = dataset_dir / 'data.yaml'

    if data_yaml.exists():
        return data_yaml

    # 尝试自动生成
    import sys
    sys.path.insert(0, str(root / 'tools'))
    from split_dataset import split_data  # type: ignore

    split_data(val_ratio=val_ratio)

    if not data_yaml.exists():
        raise FileNotFoundError(f"未生成 data.yaml: {data_yaml}")

    return data_yaml


def train(
    model: str = 'yolov8n.pt',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,  # 从4改为16，更好利用GPU
    # 默认优先使用第一张 GPU；如无 CUDA，Ultralytics 会自动回退到 CPU
    device: str | int | None = 0,
    val_ratio: float = 0.2,
):
    root = _project_root()
    data_yaml = ensure_dataset(val_ratio=val_ratio)

    print(f"训练配置 | model={model} epochs={epochs} imgsz={imgsz} batch={batch} device={device}")

    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(root / 'runs'),
        name='detect',
        exist_ok=True,
        # 添加一些有用的训练参数
        save_period=10,  # 每10个epoch保存一次
        patience=50,     # 早停耐心值
        workers=8,       # 数据加载线程数
    )

    # 复制 best.pt 到 data/models 方便管理
    models_dir = root / 'data' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics 会把权重输出到 runs/detect/train*/weights/best.pt
    # 从 results.save_dir 反推路径更稳
    try:
        save_dir = Path(results.save_dir)  # type: ignore[attr-defined]
    except Exception:
        # 兜底：按默认目录找最新一次训练
        save_dir = root / 'runs' / 'detect'

    best_pt = save_dir / 'weights' / 'best.pt'
    if best_pt.exists():
        dst = models_dir / 'yolo_best.pt'
        shutil.copyfile(best_pt, dst)
        print(f"训练完成，best.pt 已复制到: {dst}")
    else:
        print(f"训练完成，但未找到 best.pt: {best_pt}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLO detector (GPU preferred).')
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument(
        '--device',
        default='0',
        help="训练设备：'0'/'1'... 表示GPU编号；或 'cpu'。默认 0",
    )

    args = parser.parse_args()

    # Ultralytics 接受 device='cpu' 或 device=0/1/...
    dev: str | int | None
    if isinstance(args.device, str) and args.device.lower() == 'cpu':
        dev = 'cpu'
    else:
        # 允许传 '0' 这种字符串
        try:
            dev = int(args.device)
        except Exception:
            dev = args.device

    train(
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=dev,
        val_ratio=args.val_ratio,
    )
