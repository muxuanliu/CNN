import os  # 导入os模块，用于操作文件和目录
from PIL import Image  # 导入PIL库中的Image模块，用于图像处理
import numpy as np  # 导入numpy库，用于高效的数值计算

# 设置包含图像文件的目录路径
file_path = './data/medical-mnist'

# 初始化总像素计数和用于计算平均像素值的变量
total_pixels = 0
# 初始化一个形状为(3,)的零数组，用于累加每个颜色通道的归一化像素值
sum_normalized_pixed_values = np.zeros(1)

# 使用os.walk遍历指定目录及其所有子目录
for root, dirs, files in os.walk(file_path):
    # 遍历当前目录下的所有文件
    for filename in files:
        # 检查文件扩展名是否是图片格式
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 构建完整的文件路径
            image_path = os.path.join(root, filename)
            # 打开图像文件
            image = Image.open(image_path)
            # 将图像转换为numpy数组
            image_array = np.array(image)

            # 将像素值归一化到0-1区间
            normalized_image_array = image_array / 255.0

            # 累加归一化后的像素值，用于后续计算平均值
            total_pixels += normalized_image_array.size
            sum_normalized_pixed_values += np.sum(normalized_image_array, axis=(0, 1))

# 计算所有像素值的平均值
mean = sum_normalized_pixed_values[0] / total_pixels

print('Mean:', mean)

# 初始化一个形状为(3,)的零数组，用于计算每个颜色通道的方差
sum_squared_diff = np.zeros(1)

# 第二次遍历目录，计算每个像素值与平均值差的平方和
for root, dirs, files in os.walk(file_path):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            image_array = np.array(image)

            normalized_image_array = image_array / 255.0

            # 计算每个像素值与平均值差的平方
            diff = (normalized_image_array - mean) ** 2
            # 累加每个颜色通道的平方差，用于后续计算方差
            sum_squared_diff += np.sum(diff, axis=(0, 1))

# 计算方差，即平方差和除以总像素数
variance = sum_squared_diff[0] / total_pixels

print("Variance:", variance)