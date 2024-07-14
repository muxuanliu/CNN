# 导入所需的Python库
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
import torch
from torch import nn
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import GoogLeNet, Inception  # model.py中定义了LeNet模型
from tqdm import tqdm  # 导入tqdm库，用于显示进度条


# 定义数据加载和处理函数
def train_val_data_process():
    ROOT_TRAIN = r'./data/medical-mnist/train'

    normalize = transforms.Normalize(mean=[0.359], std=[0.079])
    # 定义数据集处理方法,transforms.Graysacle转为处理灰度图像即单通道
    train_transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((224,224)),transforms.ToTensor(),normalize])

    # 加载数据集
    train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)

    # 将加载的数据集分为80%的训练数据和20%的验证数据
    train_data, val_data = Data.random_split(train_data,
                                             lengths=[round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    # 为训练数据和验证数据创建DataLoader，设置批量大小为16，洗牌，2个进程加载数据
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=16,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=16,
                                     shuffle=True,
                                     num_workers=2)

    # 返回训练和验证的DataLoader
    return train_dataloader, val_dataloader



# 定义模型训练和验证过程的函数
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设置使用CUDA如果可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印使用的设备
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'当前模型训练设备为: {dev}')

    # 初始化Adam优化器和交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 将模型移动到选定的设备上
    model = model.to(device)

    # 复制模型权重用于后续更新最佳模型
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # 初始化最佳准确度

    # 初始化用于记录训练和验证过程中损失和准确度的列表
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    # 记录训练开始时间
    start_time = time.time()

    # 迭代指定的训练轮数
    for epoch in range(1, num_epochs + 1):
        # 记录每个epoch开始的时间
        since = time.time()

        # 打印分隔符和当前epoch信息
        print("-" * 10)
        print(f"Epoch: {epoch}/{num_epochs}")

        # 初始化训练和验证过程中的损失和正确预测数量
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0

        # 初始化批次计数器
        train_num = 0
        val_num = 0

        # 创建训练进度条
        progress_train_bar = tqdm(total=len(train_dataloader), desc=f'Training {epoch}', unit='batch')

        # 训练数据集的遍历
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将数据移动到相应的设备上
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 训练模型
            model.train()

            # 前向传播
            output = model(b_x)

            # 计算预测标签
            pre_label = torch.argmax(output, dim=1)

            # 计算损失
            loss = criterion(output, b_y)

            # 清空梯度
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()

            # 累加损失和正确预测数量
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_label == b_y.data)

            # 更新批次计数器
            train_num += b_x.size(0)

            # 更新训练进度条
            progress_train_bar.update(1)

        # 关闭训练进度条
        progress_train_bar.close()

        # 创建验证进度条
        progress_val_bar = tqdm(total=len(val_dataloader), desc=f'Validation {epoch}', unit='batch')

        # 验证数据集的遍历
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将数据移动到相应的设备上
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 评估模型
            model.eval()

            # 前向传播
            output = model(b_x)

            # 计算预测标签
            pre_label = torch.argmax(output, dim=1)

            # 计算损失
            loss = criterion(output, b_y)

            # 累加损失和正确预测数量
            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_label == b_y.data)

            # 更新批次计数器
            val_num += b_x.size(0)

            # 更新验证进度条
            progress_val_bar.update(1)

        # 关闭验证进度条
        progress_val_bar.close()

        # 计算并记录epoch的平均损失和准确度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        # 打印训练和验证的损失与准确度
        print(f'{epoch} Train Loss: {train_loss_all[-1]:.4f} Train Acc: {train_acc_all[-1]:.4f}')
        print(f'{epoch} Val Loss: {val_loss_all[-1]:.4f} Val Acc: {val_acc_all[-1]:.4f}')

        # 计算并打印epoch训练耗费的时间
        epoch_time_use = time.time() - since
        print(f'第 {epoch} 个 epoch 训练耗费时间: {epoch_time_use // 60:.0f}m {epoch_time_use % 60:.0f}s')

        # 若当前epoch的验证准确度为最佳，则更新最佳模型权重
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

    # 训练结束，保存最佳模型权重
    torch.save(best_model_wts, './weight/diy_best_model.pth')

    # 如果当前epoch为总epoch数，则保存最终模型权重
    if epoch == num_epochs:
        torch.save(model.state_dict(), f'./weight/diy_{num_epochs}_model.pth')

    # 将训练过程中的统计数据整理成DataFrame
    train_process = pd.DataFrame(data={
        "epoch": range(1, num_epochs + 1),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all
    })

    # 打印总训练时间
    consume_time = time.time() - start_time
    print(f'总耗时：{consume_time // 60:.0f}m {consume_time % 60:.0f}s')

    # 返回包含训练过程统计数据的DataFrame
    return train_process


# 定义绘制训练和验证过程中损失与准确度的函数
def matplot_acc_loss(train_process):
    # 创建图形和子图
    plt.figure(figsize=(12, 4))

    # 绘制训练和验证损失
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss_all"], 'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process["val_loss_all"], 'bs-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # 保存损失图像
    plt.savefig('./result_picture/diy_training_loss.png', bbox_inches='tight')

    # 绘制训练和验证准确度
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc_all"], 'ro-', label="train_acc")
    plt.plot(train_process["epoch"], train_process["val_acc_all"], 'bs-', label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    # 保存准确率曲线图
    plt.savefig('./result_picture/diy_training_accuracy.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    model = GoogLeNet(Inception)

    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=2)

    matplot_acc_loss(train_process)
