import torch
import torchvision.transforms as transforms
from PIL import Image

# 定义数据增广函数
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=100,expand=False),
    transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
    transforms.ToTensor(),
])

# 加载图片
image = Image.open('dog.jpg')

# 对图片进行增广
augmented_image = transform(image)

# 显示增广后的图片
import matplotlib.pyplot as plt
plt.imshow(augmented_image.permute(1, 2, 0))
plt.show()