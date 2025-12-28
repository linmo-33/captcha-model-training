import os
import shutil
import random

# 路径配置 - 支持从项目根目录或tools目录运行
if os.path.exists('data/raw'):
    RAW_DIR = 'data/raw'
    DATASET_DIR = 'data/dataset'
else:
    RAW_DIR = os.path.join('..', 'data', 'raw')
    DATASET_DIR = os.path.join('..', 'data', 'dataset')

def load_classes():
    """从classes.txt加载类别信息"""
    classes_file = os.path.join(RAW_DIR, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes
    else:
        print(f"警告: 未找到 {classes_file}，使用默认类别")
        return ['target']

def split_data(val_ratio=0.2):
    print(f"使用路径: RAW_DIR={RAW_DIR}, DATASET_DIR={DATASET_DIR}")
    
    # 检查源目录是否存在
    if not os.path.exists(RAW_DIR):
        print(f"错误: 源目录 {RAW_DIR} 不存在!")
        return
    
    # 加载类别信息
    classes = load_classes()
    print(f"加载了 {len(classes)} 个类别: {classes[:5]}{'...' if len(classes) > 5 else ''}")
    
    # 1. 准备目录结构
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, 'labels', split), exist_ok=True)

    # 2. 获取所有已标注的图片 (有对应的 .txt 文件才算)
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('_bg.jpg')]
    print(f"找到 {len(files)} 个 _bg.jpg 文件")
    
    labeled_files = []
    for f in files:
        txt_name = f.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(RAW_DIR, txt_name)):
            labeled_files.append(f)
    
    print(f"其中 {len(labeled_files)} 个有对应的标注文件")
    
    if len(labeled_files) == 0:
        print("错误: 没有找到任何已标注的文件!")
        return

    # 3. 随机打乱并划分
    random.shuffle(labeled_files)
    split_idx = int(len(labeled_files) * (1 - val_ratio))
    train_files = labeled_files[:split_idx]
    val_files = labeled_files[split_idx:]

    print(f"总计: {len(labeled_files)} | 训练集: {len(train_files)} | 验证集: {len(val_files)}")

    # 4. 移动/复制文件
    def copy_files(file_list, split_type):
        print(f"正在复制 {len(file_list)} 个文件到 {split_type} 集...")
        for i, filename in enumerate(file_list):
            # 源路径
            src_img = os.path.join(RAW_DIR, filename)
            src_txt = os.path.join(RAW_DIR, filename.replace('.jpg', '.txt'))
            
            # 目标路径
            dst_img = os.path.join(DATASET_DIR, 'images', split_type, filename)
            dst_txt = os.path.join(DATASET_DIR, 'labels', split_type, filename.replace('.jpg', '.txt'))
            
            shutil.copy(src_img, dst_img)
            shutil.copy(src_txt, dst_txt)
            
            if (i + 1) % 100 == 0:
                print(f"  已复制 {i + 1}/{len(file_list)} 个文件")

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    
    # 5. 生成 data.yaml
    # 使用相对路径，更便于项目迁移
    relative_dataset_path = os.path.relpath(DATASET_DIR, '.')
    yaml_content = f"""path: {relative_dataset_path} # dataset root dir
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""
    yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
        
    print(f"数据集划分完成！")
    print(f"生成的 data.yaml 文件: {yaml_path}")
    print(f"类别数量: {len(classes)}")
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")

if __name__ == '__main__':
    split_data()