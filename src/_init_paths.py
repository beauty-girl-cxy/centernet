# 用于调用本地库
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__) # 返回脚本的路径c:\Users\FX63\Desktop\CenterNet-master\src

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib') # c:\Users\FX63\Desktop\CenterNet-master\src\lib
add_path(lib_path) # 将c:\Users\FX63\Desktop\CenterNet-master\src\lib添加到PYTHON临时搜索路径
