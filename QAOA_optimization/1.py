import numpy as np
import pandas as pd

# 1. 加载数据
data_path = "./output/budget_4_layers_4_eta_6.0.npz"
data = np.load(data_path)

selections = data['selection']
values = data['value']
probabilities = data['probability']

# 2. 构建 DataFrame
df = pd.DataFrame({
    'Selection': selections,
    'Value (Loss/Energy)': values,
    'Probability': probabilities
})
df.index.name = 'Rank'

# 3. 关键：解除 pandas 打印行数限制，确保 64 行全部显示
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print("============ QAOA 全部 64 个状态的运行结果 ============")
print(df)
print("======================================================")