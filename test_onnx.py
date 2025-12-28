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
    """后处理模型输出"""
    # YOLO输出格式: [batch, num_detections, 4 + num_classes]
    # 4: x_center, y_center, width, height
    predictions = outputs[0][0]  # 移除批次维度
    
    # 过滤低置信度检测
    scores = np.max(predictions[:, 4:], axis=1)
    valid_indices = scores > conf_threshold
    
    if not np.any(valid_indices):
        return []
    
    valid_predictions = predictions[valid_indices]
    valid_scores = scores[valid_indices]
    
    # 获取类别ID
    class_ids = np.argmax(valid_predictions[:, 4:], axis=1)
    
    # 转换边界框格式 (center_x, center_y, w, h) -> (x1, y1, x2, y2)
    boxes = valid_predictions[:, :4]
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # 调整坐标到原始图像
    pad_x, pad_y = pad_offset
    x1 = (x1 * 640 - pad_x) / scale
    y1 = (y1 * 640 - pad_y) / scale
    x2 = (x2 * 640 - pad_x) / scale
    y2 = (y2 * 640 - pad_y) / scale
    
    # 限制在图像范围内
    h, w = original_shape
    x1 = np.clip(x1, 0, w)
    y1 = np.clip(y1, 0, h)
    x2 = np.clip(x2, 0, w)
    y2 = np.clip(y2, 0, h)
    
    # NMS (非极大值抑制)
    boxes_for_nms = np.column_stack([x1, y1, x2, y2])
    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(),
        valid_scores.tolist(),
        conf_threshold,
        iou_threshold
    )
    
    results = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            results.append({
                'bbox': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                'confidence': float(valid_scores[i]),
                'class_id': int(class_ids[i])
            })
    
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


def main():
    parser = argparse.ArgumentParser(description='测试 ONNX 模型')
    parser.add_argument('--model', default='data/models/yolo_best.onnx', help='ONNX 模型路径')
    parser.add_argument('--image', required=True, help='测试图像路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    
    args = parser.parse_args()
    
    try:
        test_onnx_model(args.model, args.image, args.conf)
    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    main()