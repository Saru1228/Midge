'''
读取文件函数
load files 's function
并且对于读取的文件进行列名标注
and label the datasets as paper
'''
import os
import pandas as pd
import numpy as np

COLUMNS = ["id", "x", "z", "y", "t", "vx", "vz", "vy", "ax", "az", "ay"]

def _read_one_file(file_path):
    """
    读取单个文件，优先用逗号分隔；若列数不为11则退回空白分隔。
    返回一个DataFrame（列名已设置）。
    """
    # 方案A：逗号
    df = pd.read_csv(file_path, sep=",", header=None, engine="python")
    if df.shape[1] != 11:
        # 方案B：空白
        df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != 11:
        raise ValueError(f"{os.path.basename(file_path)} 读取后列数为 {df.shape[1]}，应为11。请检查文件内容与分隔符。")
    df.columns = COLUMNS
    # 尝试转为数值（若存在无法转换的字符串，会变为NaN）
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def read_swarm_batch(folder, start=1, end=19, prefix="Ob", ext=".txt", assign_to_locals=False):
    """
    批量读取 Ob{start}.txt ~ Ob{end}.txt
    返回: dict, 形如 {"df1": DataFrame, ..., "df19": DataFrame}
    参数:
      - folder: 含有这些文件的目录
      - start, end: 文件编号范围（闭区间）
      - prefix, ext: 文件名前缀与扩展名
      - assign_to_locals: 若为 True，则把 df1..dfN 注入到调用者的局部变量（一般不建议在函数里这么做，默认False）
    """
    result = {}
    for i in range(start, end + 1):
        filename = f"{prefix}{i}{ext}"
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未找到文件: {path}")
        df = _read_one_file(path)
        result[f"df{i}"] = df

    # 可选：把 df1..dfN 注入到上层调用环境（不推荐，默认关闭）
    if assign_to_locals:
        import inspect
        frame = inspect.currentframe().f_back
        frame.f_locals.update(result)
    return result
