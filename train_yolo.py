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


def interactive_train():
    """交互式训练配置"""
    print("=" * 50)
    print("YOLO 模型训练工具")
    print("=" * 50)
    print("提示: 直接按回车使用 [默认值]\n")

    # 预训练模型
    default_model = 'yolov8n.pt'
    model_options = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    print(f"可选模型: {', '.join(model_options)}")
    model = input(f"预训练模型 [{default_model}]: ").strip()
    model = model if model else default_model

    # 训练轮数
    default_epochs = 100
    epochs_input = input(f"训练轮数 [{default_epochs}]: ").strip()
    epochs = int(epochs_input) if epochs_input else default_epochs

    # 图像尺寸
    default_imgsz = 640
    imgsz_input = input(f"图像尺寸 [{default_imgsz}]: ").strip()
    imgsz = int(imgsz_input) if imgsz_input else default_imgsz

    # 批次大小
    default_batch = 16
    batch_input = input(f"批次大小 [{default_batch}]: ").strip()
    batch = int(batch_input) if batch_input else default_batch

    # 设备选择
    default_device = '0'
    device_input = input(f"训练设备 (0/1/cpu) [{default_device}]: ").strip()
    device_input = device_input if device_input else default_device

    # 验证集比例
    default_val_ratio = 0.2
    val_ratio_input = input(f"验证集比例 [{default_val_ratio}]: ").strip()
    val_ratio = float(val_ratio_input) if val_ratio_input else default_val_ratio

    # 解析设备
    if device_input.lower() == 'cpu':
        device = 'cpu'
    else:
        try:
            device = int(device_input)
        except ValueError:
            device = device_input

    # 确认配置
    print("\n" + "=" * 50)
    print("训练配置:")
    print(f"  预训练模型: {model}")
    print(f"  训练轮数: {epochs}")
    print(f"  图像尺寸: {imgsz}")
    print(f"  批次大小: {batch}")
    print(f"  训练设备: {device}")
    print(f"  验证集比例: {val_ratio}")
    print("=" * 50)

    confirm = input("\n确认开始训练? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("已取消")
        return

    train(
        model=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        val_ratio=val_ratio,
    )


if __name__ == '__main__':
    interactive_train()
