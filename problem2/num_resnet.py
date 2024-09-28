import torch
import torchvision
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch import nn
from PIL import Image
import os
import torch.optim as optim
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def dataReady(Path: str = "datasets"):
    # 将图像转换为张量并进行归一化
    transform = transforms.Compose(
        [transforms.Resize((28,28)),
         transforms.ToTensor(),
         transforms.Grayscale(num_output_channels=1),
         transforms.Normalize((0.5,), (0.5,))]
    )

    # 下载MNIST训练集
    mnist_train_dataset = torchvision.datasets.MNIST(
        root=Path, train=True, download=True, transform=transform)
    # 下载MNIST测试集
    mnist_test_dataset = torchvision.datasets.MNIST(
        root=Path, train=False, download=True, transform=transform)
    random.seed(0)
    # 生成一个包含19000个随机索引的列表
    indices = random.sample(range(len(mnist_train_dataset)), 19000)

    mnist_train_dataset = torch.utils.data.Subset(mnist_train_dataset, indices)
    #额外的训练集和测试集
    extra_dataset_path = r'D:\code\pythonProject2\datasets\EXTRA2'
    extra_dataset = extradata(root_dir=extra_dataset_path, transform=transform)
    #划分测试集和训练集
    train_size = int(0.86*len(extra_dataset))
    test_size = len(extra_dataset)-train_size
    extra_train_dataset,extra_test_dataset = random_split(extra_dataset,[train_size,test_size])
    #合并
    combined_train_dataset = ConcatDataset([mnist_train_dataset,extra_train_dataset])
    combined_test_dataset = ConcatDataset([mnist_test_dataset,extra_test_dataset])
    print(len(combined_train_dataset))
    print(len(combined_test_dataset))
    # 加载训练集
    train_loader = DataLoader(dataset=combined_train_dataset, batch_size=64, shuffle=True)
    # 加载测试集
    test_loader = DataLoader(dataset=combined_test_dataset, batch_size=64, shuffle=True)
    return train_loader, test_loader

class extradata(Dataset): #额外寻找的数据集
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label in range(10):
            folder_path = os.path.join(root_dir,str(label))
            for filename in os.listdir(folder_path):
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    self.images.append(os.path.join(folder_path,filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = self.images[item]
        image = Image.open(image_path)
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)

        return image, label

#残差块
class Residual(nn.Module):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:  #对于残差是否要使用1x1卷积层进行变换
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y = Y + X
        return F.relu(Y)

# 残差网络
class ResNet(nn.Module):

    #开头部分
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
   #残差模块
    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)

    def __init__(self, arch, lr=0.01, num_classes=10):
        super(ResNet, self).__init__()
        self.initialized = False
        self.lr = lr
        self.num_classes = num_classes
        self.net = nn.ModuleList([self.b1()])
        for i, b in enumerate(arch):
            self.net.append(self.block(*b, first_block=(i == 0)))
        self.net.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.init_cnn(self.net)
    #初始化cnn网络模块，由于使用的是lazy
    def init_cnn(self, module):
        if isinstance(module, (nn.LazyConv2d, nn.LazyBatchNorm2d, nn.LazyLinear)):
            if hasattr(module, 'weight') and module.weight is not None:  #检查 module 是否有权重属性，并且该权重不为 None。
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                module.weight.requires_grad = True
            if hasattr(module, 'bias') and module.bias is not None: #检查 module 是否有bias属性，并且该权重不为 None。
                nn.init.constant_(module.bias, 0)
                module.bias.requires_grad = True

    def forward(self, x):
        if not self.initialized:
            # 确保在第一次前向传播时对网络层进行适当的初始化
            with torch.enable_grad():
                for layer in self.net:
                    x = layer(x)
            self.apply(self.init_cnn)
            for param in self.parameters():
                param.requires_grad = True
            self.initialized = True
        else:
            for layer in self.net:
                x = layer(x)
        return x

class ResNet18(ResNet):  #ResNet18
    def __init__(self, lr=0.01, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256),(2,512)),
                       lr=lr, num_classes=num_classes)

class ResNet50(ResNet):  #ResNet50
    def __init__(self, lr=0.01, num_classes=10):
        super().__init__(((3, 64), (4, 128), (6, 256), (3, 512)),
                         lr=lr, num_classes=num_classes)

def train_model(train_data_loader:DataLoader,model:nn.Module,optimizer,loss_fn):  #训练模型
    datanum = len(train_data_loader.dataset)
    batch_num = len(train_data_loader)
    train_loss = 0
    train_accucary = 0
    for x,y in train_data_loader: #遍历训练集
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()  #清除之前的梯度
        loss.backward()
        optimizer.step()  #根据计算的梯度更新模型参数
        train_accucary = train_accucary+(pred.argmax(1) == y).type(torch.float32).sum().item()
        train_loss = train_loss+loss.item()

    train_accucary = train_accucary/datanum
    train_loss = train_loss/batch_num
    return train_loss,train_accucary

def run_train(Path,epochs,is_save=False,is_eval=True,m=ResNet18):
    '''
    :param is_save: 是否保存模型
    :param is_eval: 是否评估模型
    :return: 返回模型
    '''
    train_loader, test_loader = dataReady(Path)
    model = m()
    model = model.to(device)
    lossf = torch.nn.CrossEntropyLoss() #损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001) #Adam优化器
    for epoch in range(epochs):
        model.train() #切换为训练模式
        train_loss, train_accuracy = train_model(train_loader, model, optimizer, lossf)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy * 100:.3f}% ")
    if is_save:
        save_path = 'models'
        model_name = get_next_it(save_path) #寻找合适的名字
        torch.save(model.state_dict(), os.path.join(save_path, model_name)) #保存模型
        print(f"Model saved as {model_name}")
    if is_eval:
        average_loss,test_accuracy = eval_model(model,test_loader)
        print(f"Test Loss:{average_loss},Accuracy:{test_accuracy*100:.3f}% ")
    return model


def get_next_it(Path='models', base_name="model"):
    # 列出文件夹中所有文件
    existing_files = os.listdir(Path)
    # 筛选出符合 model_数字.pth 格式的文件
    model_files = [f for f in existing_files if f.startswith(base_name) and f.endswith('.pth')]
    # 提取文件名中的数字部分
    model_numbers = []
    for model_file in model_files:
        try:
            # 分割 model 文件名，提取序号部分
            number = int(model_file[len(base_name) + 1:-4])
            model_numbers.append(number)
        except ValueError:
            pass

    # 找到最大的序号，新的文件将使用下一个序号
    if model_numbers:
        next_number = max(model_numbers) + 1
    else:
        next_number = 1  # 如果没有文件，使用 1 作为初始序号

    # 返回新的文件名
    return f"{base_name}_{next_number}.pth"

#加载模型
def load_model(model_name,Path = 'models',m=ResNet18):
    model = m()
    model_path = os.path.join(Path,model_name)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    return model
def eval_model(model, test_loader, loss_fn=torch.nn.CrossEntropyLoss()):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():  #评估时禁用梯度运算
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    average_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return average_loss, accuracy

#预测单张图片
def pred_img(imgPath,model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    image = Image.open(imgPath).convert('L')
    image_tensor = transform(image.resize((28, 28)))
    image_tensor = image_tensor.unsqueeze(0) #添加一个维度以模拟batch_size
    image_tensor = image_tensor.to(device)
    plt.imshow(image_tensor.cpu().squeeze().numpy(), cmap='gray')
    plt.show()  #展示图片
    with torch.no_grad():
        model.eval()  # 将模型设置为评估模式
        outputs = model(image_tensor)
    # 获取模型的预测结果
    _, result = torch.max(outputs, 1)
    print(f"识别结果: {result.item()}")

if __name__ == "__main__":
    num_classes = 10
    Path = 'datasets'
    mode = 2
    if mode == 1:
        # 从头训练模型
        model = run_train(Path, 20, is_save=True,m=ResNet50)
    else:
        model = load_model(model_name='model_19.pth', m=ResNet50)
    _, test_loader = dataReady(Path)
    average_loss, accuracy = eval_model(model, test_loader)
    print(f"Test Loss:{average_loss},Accuracy:{accuracy * 100:.3f}% ")

    img_path = r"my_wrriten_digits\3333.jpg"
    pred_img(img_path,model)

