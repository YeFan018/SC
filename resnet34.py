import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

# 加载预训练的 ResNet34 模型
resnet34 = models.resnet34(weights='IMAGENET1K_V1')
resnet34.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(file_path):
    image = Image.open(file_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批次维度
    return image

image_folder_path = r'F:\wukong_test\wukong_test'  # 图像文件夹路径
image_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

arrays = []
count = 0
for file_path in image_files:
    image_tensor = process_image(file_path)
    with torch.no_grad():
        output = resnet34(image_tensor)  # 提取图像特征
    arrays.append(output.cpu().numpy())  # 将特征添加到数组中
    count += 1
    if count % 100 == 0:
        print(f"Processed {count} images.")

# 确保数组不为空后再进行拼接
if len(arrays) > 0:
    output_array = np.concatenate(arrays, axis=0)
    np.savez('feature_extracted_img.npz', data=output_array)
    print(f"Feature extraction complete. Saved to 'feature_extracted_img.npz'.")
else:
    print("No images were processed. Please check the file paths and images.")
