"""
导出 YOLO 模型为 ONNX 格式

用于生产环境部署，ONNX 模型更轻量且不依赖 ultralytics
"""

from pathlib import Path
from ultralytics import YOLO


def export_to_onnx(
    model_path: str = 'data/models/yolo_best.pt',
    output_path: str = 'data/models/yolo_best.onnx',
    imgsz: int = 640,
    simplify: bool = True,
    dynamic: bool = False,
    half: bool = False
):
    """
    导出 YOLO 模型为 ONNX 格式
    
    Args:
        model_path: PyTorch 模型路径
        output_path: ONNX 输出路径
        imgsz: 输入图像尺寸
        simplify: 是否简化 ONNX 模型
        dynamic: 是否支持动态输入尺寸
        half: 是否使用FP16精度（减小模型大小）
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"导出 ONNX 模型...")
    model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=simplify,
        dynamic=dynamic,
        half=half,
        opset=12  # ONNX opset 版本
    )
    
    # ultralytics 会自动生成 .onnx 文件在同目录
    auto_onnx = model_path.with_suffix('.onnx')
    
    if auto_onnx.exists() and auto_onnx != output_path:
        # 移动到指定位置
        import shutil
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(auto_onnx), str(output_path))
        print(f"✓ ONNX 模型已保存: {output_path}")
    elif auto_onnx.exists():
        print(f"✓ ONNX 模型已保存: {auto_onnx}")
    else:
        print(f"✗ 导出失败")
    
    # 显示模型信息
    if output_path.exists():
        import os
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"\n模型信息:")
        print(f"  类别数量: {len(model.names)}")
        print(f"  类别列表: {list(model.names.values())}")
        print(f"  输入尺寸: {imgsz}x{imgsz}")
        print(f"  文件大小: {file_size:.2f} MB")
        print(f"  动态输入: {'是' if dynamic else '否'}")
        print(f"  FP16精度: {'是' if half else '否'}")
    elif auto_onnx.exists():
        import os
        file_size = os.path.getsize(auto_onnx) / (1024 * 1024)  # MB
        print(f"\n模型信息:")
        print(f"  类别数量: {len(model.names)}")
        print(f"  类别列表: {list(model.names.values())}")
        print(f"  输入尺寸: {imgsz}x{imgsz}")
        print(f"  文件大小: {file_size:.2f} MB")
        print(f"  动态输入: {'是' if dynamic else '否'}")
        print(f"  FP16精度: {'是' if half else '否'}")


def interactive_export():
    """交互式导出配置"""
    print("=" * 50)
    print("YOLO 模型导出 ONNX 工具")
    print("=" * 50)
    print("提示: 直接按回车使用 [默认值]\n")

    # 模型路径
    default_model = 'data/models/yolo_best.pt'
    model_path = input(f"模型路径 [{default_model}]: ").strip()
    model_path = model_path if model_path else default_model

    # 输出路径
    default_output = model_path.replace('.pt', '.onnx')
    output_path = input(f"输出路径 [{default_output}]: ").strip()
    output_path = output_path if output_path else default_output

    # 图像尺寸
    default_imgsz = 640
    imgsz_input = input(f"输入尺寸 [{default_imgsz}]: ").strip()
    imgsz = int(imgsz_input) if imgsz_input else default_imgsz

    # 是否简化
    simplify_input = input("简化模型 [Y/n]: ").strip().lower()
    simplify = simplify_input != 'n'

    # 动态输入
    dynamic_input = input("动态输入 [y/N]: ").strip().lower()
    dynamic = dynamic_input == 'y'

    # FP16 精度
    half_input = input("FP16精度 [y/N]: ").strip().lower()
    half = half_input == 'y'

    # 确认配置
    print("\n" + "=" * 50)
    print("导出配置:")
    print(f"  模型路径: {model_path}")
    print(f"  输出路径: {output_path}")
    print(f"  输入尺寸: {imgsz}")
    print(f"  简化模型: {'是' if simplify else '否'}")
    print(f"  动态输入: {'是' if dynamic else '否'}")
    print(f"  FP16精度: {'是' if half else '否'}")
    print("=" * 50)

    confirm = input("\n确认导出? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("已取消")
        return

    export_to_onnx(
        model_path=model_path,
        output_path=output_path,
        imgsz=imgsz,
        simplify=simplify,
        dynamic=dynamic,
        half=half
    )


if __name__ == '__main__':
    interactive_export()
