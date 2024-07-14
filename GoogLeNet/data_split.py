import os
import random
import shutil
from tqdm import tqdm


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

file_path = './data/medical-mnist'
flower_class = [cla for cla in os.listdir(file_path)]

train_path = f'{file_path}/train/'
mkfile(train_path)
for cla in flower_class:
    mkfile(train_path + cla)

test_path = f'{file_path}/test/'
mkfile(test_path)
for cla in flower_class:
    mkfile(test_path + cla)

# 划分训练集和测试集
split_rate = 0.1


for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))
    for index, image in tqdm(enumerate(images), unit='images', desc='spliting:'):
        if image in eval_index:
            image_path = cla_path + image
            new_path = test_path + cla
            shutil.move(image_path, new_path)

        # 其余图像保存在训练集中
        else:
            image_path = cla_path + image
            new_path = train_path + cla
            shutil.move(image_path, new_path)


print('data_split done!')