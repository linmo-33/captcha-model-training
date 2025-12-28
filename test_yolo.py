import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt


def _project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parent


def load_model(model_path: str = None) -> YOLO:
    """加载训练好的模型"""
    root = _project_root()
    
    if model_path is None:
        # 默认使用训练后的最佳模型
        model_path = root / 'data' / 'models' / 'yolo_best.pt'
        if not model_path.exists():
            # 如果没有，尝试从runs目录找最新的
            runs_dir = root / 'runs' / 'detect'
            if runs_dir.exists():
                train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('train')]
                if train_dirs:
                    latest_train = max(train_dirs, key=lambda x: x.stat().st_mtime)
                    model_path = latest_train / 'weights' / 'best.pt'
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型: {model_path}")
    return YOLO(str(model_path))


def test_single_image(model: YOLO, image_path: str, conf_threshold: float = 0.25):
    """测试单张图片"""
    print(f"测试图片: {image_path}")
    
    # 预测
    results = model(image_path, conf=conf_threshold)
    
    # 显示结果
    for r in results:
        # 打印检测结果
        if len(r.boxes) > 0:
            print(f"检测到 {len(r.boxes)} 个目标:")
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                print(f"  {i+1}. {cls_name}: {conf:.3f}")
        else:
            print("未检测到任何目标")
        
        # 保存带标注的图片
        annotated = r.plot()
        output_path = f"test_result_{Path(image_path).stem}.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"结果已保存到: {output_path}")
    
    return results


def test_directory(model: YOLO, test_dir: str, conf_threshold: float = 0.25, max_images: int = 10):
    """测试目录中的多张图片"""
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"测试目录不存在: {test_dir}")
        return
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in test_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"在 {test_dir} 中未找到图片文件")
        return
    
    # 限制测试图片数量
    image_files = image_files[:max_images]
    print(f"测试 {len(image_files)} 张图片...")
    
    results_summary = []
    
    for img_file in image_files:
        print(f"\n--- 测试: {img_file.name} ---")
        results = model(str(img_file), conf=conf_threshold)
        
        for r in results:
            detections = []
            if len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    detections.append((cls_name, conf))
                    print(f"  {cls_name}: {conf:.3f}")
            else:
                print("  未检测到目标")
            
            results_summary.append({
                'image': img_file.name,
                'detections': detections
            })
    
    return results_summary


def test_validation_set(model: YOLO, conf_threshold: float = 0.25):
    """在验证集上测试模型性能"""
    root = _project_root()
    val_dir = root / 'data' / 'dataset' / 'images' / 'val'
    
    if not val_dir.exists():
        print(f"验证集目录不存在: {val_dir}")
        return
    
    print("在验证集上评估模型...")
    
    # 使用YOLO的val方法进行评估
    data_yaml = root / 'data' / 'dataset' / 'data.yaml'
    if data_yaml.exists():
        results = model.val(data=str(data_yaml), conf=conf_threshold)
        print("验证完成！结果已保存到runs目录")
        return results
    else:
        print(f"未找到data.yaml文件: {data_yaml}")
        # 手动测试验证集
        return test_directory(model, str(val_dir), conf_threshold)


def interactive_test(model: YOLO):
    """交互式测试"""
    print("\n=== 交互式测试模式 ===")
    print("输入图片路径进行测试，输入 'quit' 退出")
    
    while True:
        image_path = input("\n请输入图片路径: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if not image_path:
            continue
            
        if not Path(image_path).exists():
            print(f"文件不存在: {image_path}")
            continue
        
        try:
            test_single_image(model, image_path)
        except Exception as e:
            print(f"测试失败: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='测试训练好的YOLO模型')
    parser.add_argument('--model', help='模型文件路径（默认使用最新训练的模型）')
    parser.add_argument('--image', help='测试单张图片')
    parser.add_argument('--dir', help='测试目录中的图片')
    parser.add_argument('--val', action='store_true', help='在验证集上测试')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互式测试')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--max-images', type=int, default=10, help='测试目录时的最大图片数')
    
    args = parser.parse_args()
    
    try:
        # 加载模型
        model = load_model(args.model)
        print(f"模型类别: {list(model.names.values())}")
        
        if args.image:
            # 测试单张图片
            test_single_image(model, args.image, args.conf)
        elif args.dir:
            # 测试目录
            test_directory(model, args.dir, args.conf, args.max_images)
        elif args.val:
            # 测试验证集
            test_validation_set(model, args.conf)
        elif args.interactive:
            # 交互式测试
            interactive_test(model)
        else:
            print("请指定测试模式:")
            print("  --image <path>     测试单张图片")
            print("  --dir <path>       测试目录")
            print("  --val              测试验证集")
            print("  --interactive      交互式测试")
            
    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    main()