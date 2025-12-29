import cv2
from pathlib import Path
from ultralytics import YOLO


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


def interactive_menu():
    """交互式测试菜单"""
    print("=" * 50)
    print("YOLO 模型测试工具")
    print("=" * 50)
    print("提示: 直接按回车使用 [默认值]\n")

    # 模型路径
    default_model = 'data/models/yolo_best.pt'
    model_path = input(f"模型路径 [{default_model}]: ").strip()
    model_path = model_path if model_path else default_model

    # 加载模型
    try:
        model = load_model(model_path if model_path != default_model else None)
        print(f"模型类别: {list(model.names.values())}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 测试模式
    print("\n测试模式:")
    print("  1. 测试单张图片")
    print("  2. 测试目录")
    print("  3. 测试验证集")
    print("  4. 交互式连续测试")
    mode = input("选择模式 [1]: ").strip()
    mode = mode if mode else '1'

    # 置信度阈值
    default_conf = 0.25
    conf_input = input(f"置信度阈值 [{default_conf}]: ").strip()
    conf = float(conf_input) if conf_input else default_conf

    if mode == '1':
        # 单张图片
        image_path = input("图片路径: ").strip()
        if not image_path:
            print("错误: 请输入图片路径")
            return

        print("\n" + "=" * 50)
        print("测试配置:")
        print(f"  模型路径: {model_path}")
        print(f"  图片路径: {image_path}")
        print(f"  置信度阈值: {conf}")
        print("=" * 50)

        confirm = input("\n确认开始测试? [Y/n]: ").strip().lower()
        if confirm == 'n':
            print("已取消")
            return

        test_single_image(model, image_path, conf)

    elif mode == '2':
        # 测试目录
        default_dir = 'data/raw'
        test_dir = input(f"测试目录 [{default_dir}]: ").strip()
        test_dir = test_dir if test_dir else default_dir

        default_max = 10
        max_input = input(f"最大图片数 [{default_max}]: ").strip()
        max_images = int(max_input) if max_input else default_max

        print("\n" + "=" * 50)
        print("测试配置:")
        print(f"  模型路径: {model_path}")
        print(f"  测试目录: {test_dir}")
        print(f"  最大图片数: {max_images}")
        print(f"  置信度阈值: {conf}")
        print("=" * 50)

        confirm = input("\n确认开始测试? [Y/n]: ").strip().lower()
        if confirm == 'n':
            print("已取消")
            return

        test_directory(model, test_dir, conf, max_images)

    elif mode == '3':
        # 测试验证集
        print("\n" + "=" * 50)
        print("测试配置:")
        print(f"  模型路径: {model_path}")
        print(f"  置信度阈值: {conf}")
        print("=" * 50)

        confirm = input("\n确认开始验证集测试? [Y/n]: ").strip().lower()
        if confirm == 'n':
            print("已取消")
            return

        test_validation_set(model, conf)

    elif mode == '4':
        # 交互式连续测试
        interactive_test(model)

    else:
        print("无效的模式选择")


if __name__ == '__main__':
    interactive_menu()