# Example: 语义缓存的预处理流程
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载文本和图像特征
text_embeddings = np.load('feature_extracted_text.npz')['data']
image_embeddings = np.load('feature_extracted_img.npz')['data']

# 计算文本之间的语义相似度矩阵
semantic_matrix_text = cosine_similarity(text_embeddings)

# 计算图像之间的语义相似度矩阵
semantic_matrix_image = cosine_similarity(image_embeddings)

# 计算文本和图像之间的相似度矩阵（如果需要）
semantic_matrix_text_image = cosine_similarity(text_embeddings, image_embeddings)

# 保存语义相似度矩阵
np.save('semantic_matrix_text.npy', semantic_matrix_text)
np.save('semantic_matrix_image.npy', semantic_matrix_image)
np.save('semantic_matrix_text_image.npy', semantic_matrix_text_image)
