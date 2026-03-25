import numpy as np

# 加载原始矩阵
filepath = r'C:\Users\11\Desktop\TNSE-code\TNSE-code\sc\semantic_matrix.npy'
matrix = np.load(filepath)
print(f"原始矩阵统计: min={matrix.min():.4f}, max={matrix.max():.4f}, mean={matrix.mean():.4f}")

# 归一化到 [0, 1]
matrix_min = matrix.min()
matrix_max = matrix.max()
if matrix_max != matrix_min:
    normalized_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
else:
    normalized_matrix = np.ones_like(matrix)
print(f"归一化后矩阵统计: min={normalized_matrix.min():.4f}, max={normalized_matrix.max():.4f}, mean={normalized_matrix.mean():.4f}")

# 保存归一化后的矩阵
np.save(filepath, normalized_matrix)
print(f"归一化矩阵已保存到: {filepath}")