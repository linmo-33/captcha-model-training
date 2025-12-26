import os
import shutil
import random

# 路径配置
RAW_DIR = os.path.join('..', 'data', 'raw')
DATASET_DIR = os.path.join('..', 'data', 'dataset')

def split_data(val_ratio=0.2):
    # 1. 准备目录结构
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, 'labels', split), exist_ok=True)

    # 2. 获取所有已标注的图片 (有对应的 .txt 文件才算)
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('_bg.jpg')]
    labeled_files = []
    for f in files:
        txt_name = f.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(RAW_DIR, txt_name)):
            labeled_files.append(f)

    # 3. 随机打乱并划分
    random.shuffle(labeled_files)
    split_idx = int(len(labeled_files) * (1 - val_ratio))
    train_files = labeled_files[:split_idx]
    val_files = labeled_files[split_idx:]

    print(f"总计: {len(labeled_files)} | 训练集: {len(train_files)} | 验证集: {len(val_files)}")

    # 4. 移动/复制文件
    def copy_files(file_list, split_type):
        for filename in file_list:
            # 源路径
            src_img = os.path.join(RAW_DIR, filename)
            src_txt = os.path.join(RAW_DIR, filename.replace('.jpg', '.txt'))
            
            # 目标路径
            dst_img = os.path.join(DATASET_DIR, 'images', split_type, filename)
            dst_txt = os.path.join(DATASET_DIR, 'labels', split_type, filename.replace('.jpg', '.txt'))
            
            shutil.copy(src_img, dst_img)
            shutil.copy(src_txt, dst_txt)

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    
    # 5. 生成 data.yaml
    yaml_content = f"""
path: {os.path.abspath(DATASET_DIR)} # dataset root dir
train: images/train
val: images/val

nc: 1
names: ['target']
"""
    with open(os.path.join(DATASET_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
        
    print("数据集划分完成！")

if __name__ == '__main__':
    split_data()