# 验证码目标检测模型训练与部署

基于 YOLOv8 的验证码图标检测系统，支持 24 种不同类型的验证码元素识别。

## 📋 目录结构

```
captcha-model-training/
├── data/
│   ├── raw/                    # 原始数据（图片+标注）
│   ├── dataset/               # 训练数据集
│   └── models/                # 训练好的模型
├── tools/                     # 数据处理工具
├── train_yolo.py             # 训练脚本
├── test_yolo.py              # PyTorch 模型测试
├── export_onnx.py            # ONNX 导出脚本
├── test_onnx.py              # ONNX 模型测试
└── requirements.txt          # 依赖包
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# ONNX 推理依赖（可选）
pip install onnxruntime  # CPU 版本
# pip install onnxruntime-gpu  # GPU 版本
```

### 2. 数据准备

将标注好的数据放入 `data/raw/` 目录：

- `*_bg.jpg` - 验证码图片
- `*_bg.txt` - YOLO 格式标注文件
- `classes.txt` - 类别名称文件

### 3. 训练模型

```bash
# 基础训练（100 epochs）
python train_yolo.py

# 自定义参数训练
python train_yolo.py --epochs 200 --batch 32 --imgsz 640
```

### 4. 导出 ONNX 模型

```bash
# 基础导出
python export_onnx.py

# 优化导出（FP16 精度，减小文件大小）
python export_onnx.py --half

# 动态输入尺寸
python export_onnx.py --dynamic
```

### 5. 模型测试

```bash
# PyTorch 模型测试
python test_yolo.py --val                    # 验证集测试
python test_yolo.py --dir data/raw          # 目录测试

# ONNX 模型测试
python test_onnx.py --dir data/raw           # 目录测试
python test_onnx.py --image test.jpg        # 单图测试
```

## 📖 详细使用说明

### 训练脚本 (`train_yolo.py`)

**功能**：训练 YOLOv8 目标检测模型

**参数说明**：

```bash
python train_yolo.py [OPTIONS]

选项:
  --model TEXT        预训练模型 [默认: yolov8n.pt]
  --epochs INTEGER    训练轮数 [默认: 100]
  --imgsz INTEGER     输入图像尺寸 [默认: 640]
  --batch INTEGER     批次大小 [默认: 16]
  --device TEXT       训练设备 [默认: 0 (GPU)]
  --val-ratio FLOAT   验证集比例 [默认: 0.2]
```

**使用示例**：

```bash
# 快速训练
python train_yolo.py

# 长时间精细训练
python train_yolo.py --model yolov8s.pt --epochs 300 --batch 32

# CPU 训练
python train_yolo.py --device cpu

# 自定义验证集比例
python train_yolo.py --val-ratio 0.15
```

**输出**：

- 训练好的模型：`data/models/yolo_best.pt`
- 训练日志：`runs/detect/train*/`
- TensorBoard 日志：可用 `tensorboard --logdir runs` 查看

---

### ONNX 导出脚本 (`export_onnx.py`)

**功能**：将 PyTorch 模型导出为 ONNX 格式，用于生产部署

**参数说明**：

```bash
python export_onnx.py [OPTIONS]

选项:
  --model TEXT        PyTorch 模型路径 [默认: data/models/yolo_best.pt]
  --output TEXT       ONNX 输出路径 [默认: data/models/yolo_best.onnx]
  --imgsz INTEGER     输入图像尺寸 [默认: 640]
  --half              使用 FP16 精度（减小模型大小）
  --dynamic           支持动态输入尺寸
  --no-simplify       不简化 ONNX 模型
```

**使用示例**：

```bash
# 基础导出
python export_onnx.py

# 优化导出（推荐用于生产）
python export_onnx.py --half

# 动态尺寸导出
python export_onnx.py --dynamic --imgsz 640

# 自定义路径
python export_onnx.py --model custom_model.pt --output custom_model.onnx
```

**输出信息**：

- 模型文件大小
- 类别数量和名称
- 输入尺寸信息
- 优化选项状态

---

### PyTorch 模型测试 (`test_yolo.py`)

**功能**：测试训练好的 PyTorch 模型性能

**参数说明**：

```bash
python test_yolo.py [OPTIONS]

选项:
  --model TEXT           模型路径 [自动检测最新模型]
  --image TEXT           测试单张图片
  --dir TEXT             测试目录中的图片
  --val                  在验证集上测试
  --interactive          交互式测试模式
  --conf FLOAT           置信度阈值 [默认: 0.25]
  --max-images INTEGER   测试目录时的最大图片数 [默认: 10]
```

**使用示例**：

```bash
# 验证集性能评估
python test_yolo.py --val

# 测试目录中的图片
python test_yolo.py --dir data/raw --max-images 5

# 测试单张图片
python test_yolo.py --image data/raw/test.jpg

# 交互式测试
python test_yolo.py --interactive

# 调整置信度阈值
python test_yolo.py --dir data/raw --conf 0.5
```

**输出**：

- 检测结果统计
- 每个目标的类别和置信度
- 带标注的结果图片：`test_result_*.jpg`

---

### ONNX 模型测试 (`test_onnx.py`)

**功能**：测试 ONNX 模型，纯推理环境，不依赖 ultralytics

**参数说明**：

```bash
python test_onnx.py [OPTIONS]

选项:
  --model TEXT           ONNX 模型路径 [默认: data/models/yolo_best.onnx]
  --image TEXT           测试单张图片
  --dir TEXT             测试目录中的图片
  --conf FLOAT           置信度阈值 [默认: 0.25]
  --iou FLOAT            NMS IoU 阈值 [默认: 0.45]
  --max-images INTEGER   测试目录时的最大图片数 [默认: 10]
```

**使用示例**：

```bash
# 测试目录中的图片
python test_onnx.py --dir data/raw --max-images 5

# 测试单张图片
python test_onnx.py --image data/raw/test.jpg

# 调整检测参数
python test_onnx.py --dir data/raw --conf 0.3 --iou 0.4

# 批量测试
python test_onnx.py --dir data/raw --max-images 20
```

**输出**：

- 推理时间统计
- 检测结果详情
- 性能统计信息
- 带标注的结果图片：`onnx_result_*.jpg`

---

## 🎯 支持的验证码类别

模型支持识别以下 24 种验证码元素：

**数字类**：`num_0`, `num_1`, `num_2`, `num_4`, `num_5`, `num_7`, `num_8`, `num_9`

**图标类**：`local`, `github`, `player`, `cat`, `file`

**表情类**：`haha`, `smirk`, `naughty`, `surprise`, `smile`

**其他类**：`polyline`, `report`, `castle0`, `castle1`, `castle2`, `house`

## 📊 性能指标

训练完成后的模型性能：

- **精确率 (Precision)**：99.2%
- **召回率 (Recall)**：99.3%
- **mAP@0.5**：99.1%
- **mAP@0.5:0.95**：82.9%

## 🔧 常见问题

### Q: 训练时显示 CUDA 不可用？

A: 检查 PyTorch CUDA 安装：`python -c "import torch; print(torch.cuda.is_available())"`

### Q: ONNX 模型检测效果不如 PyTorch？

A: 尝试调整参数：`--conf 0.2 --iou 0.3`

### Q: 如何添加新的验证码类别？

A: 更新 `data/raw/classes.txt` 文件，重新标注数据并训练

### Q: 模型文件太大？

A: 使用 FP16 导出：`python export_onnx.py --half`

## 🚀 生产部署建议

1. **使用 ONNX 模型**：更快的推理速度，更小的依赖
2. **启用 FP16**：减小模型大小，提升速度
3. **调整置信度**：根据实际需求平衡精度和召回
4. **批量处理**：提高吞吐量

## 📝 更新日志

- **v1.0**：基础 YOLOv8 训练和测试功能
- **v1.1**：添加 ONNX 导出和测试支持
- **v1.2**：优化 ONNX 后处理，提升检测精度
- **v1.3**：完善文档和使用说明

## 📄 许可证

MIT License - 详见 LICENSE 文件
