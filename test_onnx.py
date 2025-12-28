"""
测试 ONNX 模型推理

使用 ONNXRuntime 进行推理，不依赖 ultralytics
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import argparse


def load_onnx_model(model_path: str):
    """加载 ONNX 模型"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")
    
    # 创建推理会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    # 获取输入输出信息
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()
    
    print(f"模型加载成功: {model_path}")
    print(f"输入形状: {input_info.shape}")
    print(f"输入类型: {input_info.type}")
    print(f"输出数量: {len(output_info)}")
    
    return session, input_info, output_info


def preprocess_image(image_path: str, input_size: tuple = (640, 640)):
    """预处理图像"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    original_shape = image.shape[:2]  # (H, W)
    
    # 调整大小并保持宽高比
    h, w = original_shape
    target_h, target_w = input_size
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 调整大小
    resized = cv2.resize(image, (new_w, new_h))
    
    # 创建填充图像
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    
    # 计算填充位置
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    # 放置调整后的图像
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    # 转换为模型输入格式
    # BGR -> RGB
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    
    # HWC -> CHW
    padded = padded.transpose(2, 0, 1)
    
    # 归一化到 [0, 1]
    padded = padded.astype(np.float32) / 255.0
    
    # 添加批次维度
    padded = np.expand_dims(padded, axis=0)
    
    return padded, scale, (pad_x, pad_y), original_shape


def postprocess_outputs(outputs, scale, pad_offset, original_shape, conf_threshold=0.25, iou_threshold=0.45):
    """后处理模型输出 - 优化版本，匹配 PyTorch 效果"""
    
    # ONNX 输出格式: [batch, num_features, num_anchors]
    predictions = outputs[0]  # [1, 28, 8400]
    
    # 转置为 [batch, num_anchors, num_features]
    predictions = predictions.transpose(0, 2, 1)  # [1, 8400, 28]
    predictions = predictions[0]  # 移除批次维度 [8400, 28]
    
    # 分离边界框和类别预测
    boxes = predictions[:, :4]  # [8400, 4] - x_center, y_center, width, height
    class_scores = predictions[:, 4:]  # [8400, 24] - 类别分数
    
    # 应用 sigmoid 激活（如果模型输出没有激活）
    # class_scores = 1 / (1 + np.exp(-class_scores))  # sigmoid
    
    # 获取每个检测的最大类别分数和对应的类别ID
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    # 确保类别ID在有效范围内
    class_ids = np.clip(class_ids, 0, 23)
    
    # 使用更低的初始阈值来保留更多候选
    initial_threshold = max(0.1, conf_threshold * 0.5)
    valid_indices = max_scores > initial_threshold
    
    if not np.any(valid_indices):
        return []
    
    # 筛选有效检测
    valid_boxes = boxes[valid_indices]
    valid_scores = max_scores[valid_indices]
    valid_class_ids = class_ids[valid_indices]
    
    # 转换边界框格式
    x_center, y_center, width, height = valid_boxes[:, 0], valid_boxes[:, 1], valid_boxes[:, 2], valid_boxes[:, 3]
    
    # 转换为 x1, y1, x2, y2 格式
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # 调整坐标到原始图像尺寸
    pad_x, pad_y = pad_offset
    
    # 坐标调整
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    
    # 限制在原始图像范围内
    h, w = original_shape
    x1 = np.clip(x1, 0, w)
    y1 = np.clip(y1, 0, h)
    x2 = np.clip(x2, 0, w)
    y2 = np.clip(y2, 0, h)
    
    # 更宽松的面积过滤
    areas = (x2 - x1) * (y2 - y1)
    min_area = 50  # 降低最小面积要求
    max_area = w * h * 0.9  # 提高最大面积限制
    valid_area_indices = (areas > min_area) & (areas < max_area)
    
    if not np.any(valid_area_indices):
        return []
    
    x1 = x1[valid_area_indices]
    y1 = y1[valid_area_indices]
    x2 = x2[valid_area_indices]
    y2 = y2[valid_area_indices]
    valid_scores = valid_scores[valid_area_indices]
    valid_class_ids = valid_class_ids[valid_area_indices]
    
    # 按类别分组进行 NMS（避免不同类别之间的抑制）
    results = []
    unique_classes = np.unique(valid_class_ids)
    
    for class_id in unique_classes:
        class_mask = valid_class_ids == class_id
        if not np.any(class_mask):
            continue
            
        class_boxes = np.column_stack([x1[class_mask], y1[class_mask], x2[class_mask], y2[class_mask]])
        class_scores = valid_scores[class_mask]
        
        # 对每个类别单独应用 NMS
        indices = cv2.dnn.NMSBoxes(
            class_boxes.tolist(),
            class_scores.tolist(),
            conf_threshold,  # 使用原始置信度阈值
            iou_threshold
        )
        
        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            else:
                indices = [indices] if not isinstance(indices, list) else indices
            
            class_indices = np.where(class_mask)[0]
            for i in indices:
                original_idx = class_indices[i]
                if valid_scores[original_idx] >= conf_threshold:  # 最终置信度检查
                    results.append({
                        'bbox': [int(x1[original_idx]), int(y1[original_idx]), int(x2[original_idx]), int(y2[original_idx])],
                        'confidence': float(valid_scores[original_idx]),
                        'class_id': int(class_id)
                    })
    
    # 按置信度排序
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return results


def load_class_names(classes_file: str = 'data/raw/classes.txt'):
    """加载类别名称"""
    if Path(classes_file).exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    else:
        return [f'class_{i}' for i in range(100)]  # 默认类别名


def draw_detections(image_path: str, detections: list, class_names: list, output_path: str = None):
    """绘制检测结果"""
    image = cv2.imread(image_path)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_id = det['class_id']
        
        # 获取类别名称
        class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        label = f'{class_name}: {conf:.3f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"结果已保存到: {output_path}")
    else:
        cv2.imshow('Detection Results', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_onnx_model(model_path: str, image_path: str, conf_threshold: float = 0.25):
    """测试 ONNX 模型"""
    # 加载模型
    session, input_info, output_info = load_onnx_model(model_path)
    
    # 获取输入尺寸
    input_shape = input_info.shape
    if len(input_shape) == 4:  # [batch, channels, height, width]
        input_size = (input_shape[2], input_shape[3])
    else:
        input_size = (640, 640)  # 默认尺寸
    
    # 预处理图像
    print(f"预处理图像: {image_path}")
    input_data, scale, pad_offset, original_shape = preprocess_image(image_path, input_size)
    
    # 推理
    print("执行推理...")
    input_name = input_info.name
    outputs = session.run(None, {input_name: input_data})
    
    # 后处理
    print("后处理结果...")
    detections = postprocess_outputs(outputs, scale, pad_offset, original_shape, conf_threshold)
    
    # 加载类别名称
    class_names = load_class_names()
    
    # 显示结果
    print(f"\n检测结果:")
    if detections:
        print(f"检测到 {len(detections)} 个目标:")
        for i, det in enumerate(detections):
            class_name = class_names[det['class_id']] if det['class_id'] < len(class_names) else f"class_{det['class_id']}"
            print(f"  {i+1}. {class_name}: {det['confidence']:.3f}")
        
        # 绘制结果
        output_path = f"onnx_result_{Path(image_path).stem}.jpg"
        draw_detections(image_path, detections, class_names, output_path)
    else:
        print("未检测到任何目标")
    
    return detections


def test_directory(model_path: str, test_dir: str, conf_threshold: float = 0.25, max_images: int = 10):
    """批量测试目录中的图片"""
    # 加载模型
    session, input_info, output_info = load_onnx_model(model_path)
    
    # 加载类别名称
    class_names = load_class_names()
    
    # 获取图片文件
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"测试目录不存在: {test_dir}")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in test_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"在 {test_dir} 中未找到图片文件")
        return
    
    # 限制测试数量
    image_files = image_files[:max_images]
    print(f"\n开始测试 {len(image_files)} 张图片...")
    
    total_time = 0
    total_detections = 0
    results_summary = []
    
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 测试: {img_file.name}")
        
        try:
            detections, inference_time = test_single_image_core(
                session, input_info, str(img_file), class_names, conf_threshold
            )
            
            total_time += inference_time
            total_detections += len(detections)
            
            print(f"  推理时间: {inference_time*1000:.1f}ms")
            
            if detections:
                print(f"  检测到 {len(detections)} 个目标:")
                for det in detections:
                    class_name = class_names[det['class_id']] if det['class_id'] < len(class_names) else f"class_{det['class_id']}"
                    print(f"    {class_name}: {det['confidence']:.3f}")
                
                # 保存结果图片
                output_path = f"onnx_result_{img_file.stem}.jpg"
                draw_detections(str(img_file), detections, class_names, output_path)
            else:
                print("  未检测到任何目标")
            
            results_summary.append({
                'image': img_file.name,
                'detections': len(detections),
                'inference_time': inference_time
            })
            
        except Exception as e:
            print(f"  错误: {e}")
    
    # 统计信息
    print(f"\n=== 测试完成 ===")
    print(f"总图片数: {len(image_files)}")
    print(f"总检测数: {total_detections}")
    print(f"平均推理时间: {total_time/len(image_files)*1000:.1f}ms")
    print(f"总用时: {total_time:.2f}s")
    
    return results_summary


def test_single_image_core(session, input_info, image_path: str, class_names: list, conf_threshold: float = 0.25):
    """测试单张图片的核心函数（返回检测结果和推理时间）"""
    import time
    
    # 获取输入尺寸
    input_shape = input_info.shape
    if len(input_shape) == 4:
        input_size = (input_shape[2], input_shape[3])
    else:
        input_size = (640, 640)
    
    # 预处理
    input_data, scale, pad_offset, original_shape = preprocess_image(image_path, input_size)
    
    # 推理
    start_time = time.time()
    input_name = input_info.name
    outputs = session.run(None, {input_name: input_data})
    inference_time = time.time() - start_time
    
    # 后处理
    detections = postprocess_outputs(outputs, scale, pad_offset, original_shape, conf_threshold)
    
    return detections, inference_time


def main():
    parser = argparse.ArgumentParser(description='测试 ONNX 模型')
    parser.add_argument('--model', default='data/models/yolo_best.onnx', help='ONNX 模型路径')
    parser.add_argument('--image', help='测试单张图片')
    parser.add_argument('--dir', help='测试目录中的图片')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--max-images', type=int, default=10, help='测试目录时的最大图片数')
    
    args = parser.parse_args()
    
    try:
        if args.image:
            # 测试单张图片
            test_onnx_model(args.model, args.image, args.conf)
        elif args.dir:
            # 测试目录
            test_directory(args.model, args.dir, args.conf, args.max_images)
        else:
            print("请指定测试模式:")
            print("  --image <path>     测试单张图片")
            print("  --dir <path>       测试目录")
            print("\n示例:")
            print("  python test_onnx.py --image data/raw/test.jpg")
            print("  python test_onnx.py --dir data/raw --max-images 5")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    main()