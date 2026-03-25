# Bert_Text.py: 提取文本特征
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import torch

filename = r"F:\wukong_test\wukong_test\wukong_test.csv"
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese').to('cuda')
data = pd.read_excel(filename)
text = data['text'].tolist()

batch_size = 256
num_batches = (len(text) + batch_size - 1) // batch_size

all_embeddings = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    batch_text = text[start_idx:end_idx]
    inputs = tokenizer(batch_text, return_tensors='pt', padding=True, truncation=True, max_length=64).to('cuda')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    batch_embeddings = embeddings.detach().cpu().numpy().astype(np.float64)
    all_embeddings.append(batch_embeddings)

# 保存嵌入向量
all_embeddings = np.concatenate(all_embeddings, axis=0)
np.savez('sc_extracted_text.npz', data=all_embeddings)
