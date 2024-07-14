import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
# t代表test


def t_data_process():
    test_data = FashionMNIST(root="./data",
                             train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)

    return test_dataloader


def t_model_process(model, test_dataloader):
    if model is not None:
        print('Successfully loaded the model.')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0
    all_preds = []  # 存储所有预测标签
    all_labels = []  # 存储所有实际标签

    # 只进行前向传播，不计算梯度
    with torch.no_grad():
        progress_train_bar = tqdm(total=len(test_dataloader), desc=f'Testing  ', unit='unit')

        for test_x, test_y in test_dataloader:

            test_x = test_x.to(device)
            test_y = test_y.to(device)

            # 设置模型为验证模式
            model.eval()
            # 前向传播得到一个batch的结果
            output = model(test_x)
            # 查找最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)

            # 收集预测和实际标签
            all_preds.extend(pre_lab.tolist())
            all_labels.extend(test_y.tolist())

            # 计算准确率
            test_corrects += torch.sum(pre_lab == test_y.data)

            # 将所有的测试样本进行累加
            test_num += test_x.size(0)

            # 更新训练进度条
            progress_train_bar.update(1)

        # 关闭训练进度条
        progress_train_bar.close()

    # 计算准确率
    test_acc = test_corrects.double().item() / test_num
    print(f'测试的准确率：{test_acc}')

    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig('./result_picture/Confusion_Matrix.png', bbox_inches='tight')



if __name__=="__main__":
    # 加载模型
    model = LeNet()

    print('loading model')
    # 加载权重
    model.load_state_dict(torch.load('D:/Pycharm/deepl/LeNet/weight/best_model.pth'))

    # 加载测试数据
    test_dataloader = t_data_process()

    # 加载模型测试的函数
    t_model_process(model,test_dataloader)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    classes = ['T-shirt/top','Trouser','Pullover','Dress','coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()

            output = model(b_x)
            pre_lab = torch.argmax(output,dim=1)
            result = pre_lab.item()
            label = b_y.item()

            print(f'预测值：{classes[result]}',"-----------",f'真实值：{classes[label]}')





