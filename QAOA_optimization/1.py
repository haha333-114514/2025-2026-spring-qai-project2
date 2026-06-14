import os
import glob
import numpy as np
import pandas as pd

# 显示全部行
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# 获取所有 npz 文件
npz_files = sorted(glob.glob("./output/*.npz"))

print(f"共找到 {len(npz_files)} 个 npz 文件\n")

for file_path in npz_files:
    print("\n" + "=" * 80)
    print(f"文件: {os.path.basename(file_path)}")

    try:
        data = np.load(file_path)

        selections = data['selection']
        values = data['value']
        probabilities = data['probability']

        df = pd.DataFrame({
            'Selection': selections,
            'Value (Loss/Energy)': values,
            'Probability': probabilities
        })
        df.index.name = 'Rank'

        print(df)

    except Exception as e:
        print(f"读取失败: {e}")

print("\n全部文件读取完成")