import matplotlib.pyplot as plt

# ==========================
# 数据1
# ==========================
x1 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
y1 = [77,74,68,60,44,28,14,4,1,0]

# ==========================
# 数据2（可选）
# ==========================
x2 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,100,110,120,130,140,150,160,170]
y2 = [46,42,48,44,40,40,42,40,52,58,46,50,54,58,62,58,58,60]

# 创建画布
plt.figure(figsize=(8, 6))

# # 第一条线
# plt.scatter(x1, y1, label='Theoretical Values')
# plt.plot(x1, y1)

# 第二条线（如果不需要可以删掉）
plt.scatter(x2, y2, label='Experimental Values')
plt.plot(x2, y2)

# 图标题
plt.title("Curve of the current with the degree")

# 坐标轴名称
plt.xlabel("degree/deg")
plt.ylabel("current/uA")

# 网格
plt.grid(True)

# 图例
plt.legend()

# 自动调整边距
plt.tight_layout()

# 保存图片
plt.savefig("result.png", dpi=300)

# 显示
plt.show()