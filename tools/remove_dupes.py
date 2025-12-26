import os
import imagehash
from PIL import Image
from tqdm import tqdm

# 始终以脚本所在目录为基准，避免从不同工作目录运行导致路径跑偏
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
RAW_DIR = os.path.join(_PROJECT_ROOT, 'data', 'raw')

def remove_duplicates():
    print("开始去重...")
    hashes = {}
    duplicates = []
    
    # 获取所有背景图 (_bg.jpg)
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('_bg.jpg')]
    
    for filename in tqdm(files):
        path = os.path.join(RAW_DIR, filename)
        try:
            with Image.open(path) as img:
                # 计算感知哈希
                h = str(imagehash.phash(img))
                
                if h in hashes:
                    duplicates.append(filename)
                    # 同时记录对应的 icon 文件，以便一起删除
                    icon_name = filename.replace('_bg.jpg', '_icon.png')
                    duplicates.append(icon_name)
                else:
                    hashes[h] = filename
        except Exception as e:
            print(f"无法读取 {filename}: {e}")

    print(f"发现 {len(duplicates)//2} 组重复数据，正在删除...")
    
    for file in duplicates:
        file_path = os.path.join(RAW_DIR, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            
    print(f"去重完成，剩余 {len(hashes)} 组数据。")

if __name__ == '__main__':
    remove_duplicates()