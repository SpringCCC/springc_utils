import os
import random
from pathlib import Path  
import shutil

def count_files(startpath, suffix=None):  
    total = 0  
    for path in Path(startpath).rglob('*'):
        if path.is_file():
            if suffix is None:
                total += 1
            else:
                if path.name.endswith(suffix):
                    total += 1
    return total  
  

def get_dst_dir(root:str, jpg_path:str, dst_root:str):
    # root:保存jpg的最外层目录
    # jpg_path：某个图像的完整路径，包含root
    # dst_root：希望保存路径的，最外层路径
    # 返回：dst_root + root下面到jpg的中间文件夹路径，  root下面到jpg的中间文件夹路径
    # 换句话说，就是，返回的dst就是直接可以存储的路径的最深的文件夹路径，后面加上文件名就可以存储了，且保持了与root相同的目录结构
    assert root in jpg_path, print("图像数据路径有误, 不在root中")
    mid_dir = os.path.dirname(jpg_path).replace(root, "").lstrip("/")
    dst = os.path.join(dst_root, mid_dir)
    os.makedirs(dst, exist_ok=True)
    return dst, mid_dir


def list_all_suffix_files(root_path, rate=10, suffix=".jpg"):
    #获取root目录下，所有的后缀为suffix的文件完整路径
    files = []
    for path in Path(root_path).rglob('*'):
        if path.is_file():
            if suffix is None:
                total += 1
            else:
                if path.name.endswith(suffix):
                    if random.random()>rate:continue
                    files.append(str(path))
    return files  


def checkroot(root):
    os.makedirs(root, exist_ok=True)
    shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)