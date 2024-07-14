from PIL import Image
import os
import matplotlib.pyplot as plt

# 设置文件夹路径
file_path = './data/medical-mnist/train/AbdomenCT'

# 初始化计数器
rgb_images = 0
single_channel_images = 0
i = 0
# 遍历文件夹中的所有文件
for filename in os.listdir(file_path):
    # 检查文件扩展名，确保是图像文件
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(file_path, filename)
        with Image.open(image_path) as img:
            # 检查图像的通道数
            if img.mode == 'RGB':
                rgb_images += 1
            elif img.mode == 'L':
                single_channel_images += 1

# 计算总数
total_images = rgb_images + single_channel_images

# 打印结果
print(f"Total images: {total_images}")
print(f"RGB images: {rgb_images} ({100 * rgb_images / total_images:.2f}%)")
print(f"Single channel images: {single_channel_images} ({100 * single_channel_images / total_images:.2f}%)")

# 可视化结果
labels = 'RGB Images', 'Single Channel Images'
sizes = [rgb_images, single_channel_images]
colors = ['gold', 'lightcoral']

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Image Channels in Folder')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# from PIL import Image
#
# # 打开图像文件
# file_path = 'D:/Pycharm/deepl/CNN/GoogLeNet/data/medical-mnist/train/AbdomenCT/000000.jpeg'
# image = Image.open(file_path)
#
# # 检查图像的模式
# if image.mode == 'RGB':
#     channels = 3
# elif image.mode == 'L':
#     channels = 1  # 灰度图像
# elif image.mode == 'CMYK':
#     channels = 4
# elif image.mode == 'RGBA':
#     channels = 4  # 包括透明度通道
# else:
#     # 其他模式，如'P'（调色板图像）等
#     channels = 'Unknown'
#
# print(f'The image has {channels} channels.')